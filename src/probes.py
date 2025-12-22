"""
Linear probe training for detecting latent trust variable.

Trains probes on hidden states to predict trust condition,
then checks correlation with update behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List, Dict, Tuple, Optional
import pickle

from dataset import Conversation
from model_runner import ModelOutput
from metrics import ConversationMetrics


class TrustProbe:
    """Linear probe for detecting trust from hidden states."""
    
    def __init__(self, hidden_dim: int):
        """
        Initialize probe.
        
        Args:
            hidden_dim: Dimension of hidden states
        """
        self.hidden_dim = hidden_dim
        self.probe = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        self.layer_name = None
    
    def train(
        self,
        hidden_states: np.ndarray,
        trust_labels: np.ndarray,
        layer_name: str = "unknown"
    ):
        """
        Train probe on hidden states.
        
        Args:
            hidden_states: Array of shape [n_samples, hidden_dim]
            trust_labels: Array of shape [n_samples] with 0 (low) or 1 (high)
            layer_name: Name of layer these states came from
        """
        self.probe.fit(hidden_states, trust_labels)
        self.is_trained = True
        self.layer_name = layer_name
    
    def predict(self, hidden_states: np.ndarray) -> np.ndarray:
        """Predict trust scores (0-1, higher = more trust)."""
        if not self.is_trained:
            raise ValueError("Probe not trained yet")
        return self.probe.predict_proba(hidden_states)[:, 1]  # Probability of high trust
    
    def evaluate(
        self,
        hidden_states: np.ndarray,
        trust_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate probe performance."""
        predictions = self.predict(hidden_states)
        pred_binary = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(trust_labels, pred_binary)
        auc = roc_auc_score(trust_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "auc": auc,
            "layer": self.layer_name
        }
    
    def save(self, path: str):
        """Save probe to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                "probe": self.probe,
                "hidden_dim": self.hidden_dim,
                "layer_name": self.layer_name
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load probe from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        probe = cls(data["hidden_dim"])
        probe.probe = data["probe"]
        probe.is_trained = True
        probe.layer_name = data["layer_name"]
        return probe


class ProbeTrainer:
    """Trains probes on model hidden states."""
    
    def __init__(self):
        self.probes = {}  # layer_name -> TrustProbe
    
    def extract_hidden_states(
        self,
        model_outputs: List[ModelOutput],
        layer_name: str
    ) -> np.ndarray:
        """
        Extract hidden states from model outputs for a specific layer.
        
        Args:
            model_outputs: List of model outputs
            layer_name: Name of layer to extract
        
        Returns:
            Array of shape [n_samples, hidden_dim]
        """
        states = []
        for output in model_outputs:
            if layer_name in output.hidden_states:
                state = output.hidden_states[layer_name].numpy()
                states.append(state)
            else:
                raise ValueError(f"Layer {layer_name} not found in output")
        
        return np.array(states)
    
    def extract_trust_labels(
        self,
        conversations: List[Conversation]
    ) -> np.ndarray:
        """
        Extract trust labels from conversations.
        
        Returns:
            Array of 0s (low_trust) and 1s (high_trust)
        """
        return np.array([
            1 if conv.condition == "high_trust" else 0
            for conv in conversations
        ])
    
    def train_probes(
        self,
        model_outputs: List[ModelOutput],
        conversations: List[Conversation],
        layer_names: List[str],
        train_split: float = 0.8
    ) -> Dict[str, Dict[str, float]]:
        """
        Train probes for multiple layers.
        
        Args:
            model_outputs: Model outputs with hidden states
            conversations: Corresponding conversations
            layer_names: Layers to train probes for
            train_split: Fraction to use for training
        
        Returns:
            Dictionary mapping layer_name to evaluation metrics
        """
        # Extract trust labels
        trust_labels = self.extract_trust_labels(conversations)
        
        # Split train/test
        n_train = int(len(model_outputs) * train_split)
        train_outputs = model_outputs[:n_train]
        test_outputs = model_outputs[n_train:]
        train_labels = trust_labels[:n_train]
        test_labels = trust_labels[n_train:]
        
        results = {}
        
        for layer_name in layer_names:
            try:
                # Extract hidden states
                train_states = self.extract_hidden_states(train_outputs, layer_name)
                test_states = self.extract_hidden_states(test_outputs, layer_name)
                
                # Get hidden dimension
                hidden_dim = train_states.shape[1]
                
                # Train probe
                probe = TrustProbe(hidden_dim)
                probe.train(train_states, train_labels, layer_name)
                
                # Evaluate
                train_metrics = probe.evaluate(train_states, train_labels)
                test_metrics = probe.evaluate(test_states, test_labels)
                
                # Store
                self.probes[layer_name] = probe
                results[layer_name] = {
                    "train_accuracy": train_metrics["accuracy"],
                    "train_auc": train_metrics["auc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_auc": test_metrics["auc"]
                }
                
            except Exception as e:
                print(f"Error training probe for {layer_name}: {e}")
                results[layer_name] = {"error": str(e)}
        
        return results
    
    def correlate_with_behavior(
        self,
        model_outputs: List[ModelOutput],
        metrics: List[ConversationMetrics],
        layer_name: str
    ) -> Dict[str, float]:
        """
        Check correlation between probe scores and update behavior.
        
        Args:
            model_outputs: Model outputs
            metrics: Corresponding metrics
            layer_name: Layer to use probe from
        
        Returns:
            Dictionary with correlation metrics
        """
        if layer_name not in self.probes:
            raise ValueError(f"No probe trained for layer {layer_name}")
        
        probe = self.probes[layer_name]
        
        # Get probe scores
        hidden_states = self.extract_hidden_states(model_outputs, layer_name)
        probe_scores = probe.predict(hidden_states)
        
        # Get update rates
        update_rates = np.array([m.update_rate for m in metrics])
        
        # Compute correlation
        correlation = np.corrcoef(probe_scores, update_rates)[0, 1]
        
        # Also check by condition
        high_trust_mask = np.array([
            conv.condition == "high_trust"
            for conv in [m.conversation for m in metrics]
        ])
        
        high_trust_scores = probe_scores[high_trust_mask]
        low_trust_scores = probe_scores[~high_trust_mask]
        high_trust_ur = update_rates[high_trust_mask]
        low_trust_ur = update_rates[~high_trust_mask]
        
        return {
            "overall_correlation": correlation,
            "high_trust_mean_score": high_trust_scores.mean(),
            "low_trust_mean_score": low_trust_scores.mean(),
            "high_trust_mean_ur": high_trust_ur.mean(),
            "low_trust_mean_ur": low_trust_ur.mean(),
            "score_ur_correlation": correlation
        }


if __name__ == "__main__":
    # Example usage
    print("Probe training module loaded.")
    print("Use ProbeTrainer to train probes on hidden states.")
    print("Then use correlate_with_behavior to check if probe scores predict update rates.")

