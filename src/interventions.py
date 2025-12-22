"""
Activation patching and steering for causal tests.

Implements:
- Activation patching: Patch activations from one condition into another
- Steering: Add/subtract steering vectors during generation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

from dataset import Conversation
from model_runner import HookedModel, ModelOutput
from metrics import ConversationMetrics
from probes import TrustProbe


class ActivationPatcher:
    """Performs activation patching between conditions."""
    
    def __init__(self, model: HookedModel):
        self.model = model
        self.patched_activations = {}
    
    def patch_activations(
        self,
        source_output: ModelOutput,
        target_output: ModelOutput,
        layer_name: str,
        token_position: int
    ) -> ModelOutput:
        """
        Patch activations from source into target at specified position.
        
        Args:
            source_output: Output from high-trust condition (donor)
            target_output: Output from low-trust condition (recipient)
            layer_name: Layer to patch
            token_position: Token position to patch at
        
        Returns:
            New ModelOutput with patched activations
        """
        # This is a simplified version - full implementation would need
        # to actually run the model with patched activations
        
        # In practice, you'd need to:
        # 1. Run target conversation up to token_position
        # 2. Replace activations at that position with source activations
        # 3. Continue generation from that point
        
        # For now, return a placeholder
        # Full implementation would require custom forward pass
        return target_output
    
    def run_patching_experiment(
        self,
        high_trust_conv: Conversation,
        low_trust_conv: Conversation,
        layer_name: str,
        token_position: str = "last_user_token"
    ) -> Tuple[ModelOutput, ModelOutput, ModelOutput]:
        """
        Run full patching experiment.
        
        Returns:
            (high_trust_output, low_trust_output, patched_output)
        """
        # Run both conditions
        high_output = self.model.run_conversation(high_trust_conv)
        low_output = self.model.run_conversation(low_trust_conv)
        
        # Patch
        patched_output = self.patch_activations(
            high_output, low_output, layer_name, token_position
        )
        
        return high_output, low_output, patched_output


class SteeringVector:
    """Represents a steering vector for activation steering."""
    
    def __init__(self, vector: torch.Tensor, layer_name: str):
        """
        Initialize steering vector.
        
        Args:
            vector: Steering vector [hidden_dim]
            layer_name: Layer this vector applies to
        """
        self.vector = vector
        self.layer_name = layer_name
    
    @classmethod
    def from_probe(cls, probe: TrustProbe, layer_name: str) -> 'SteeringVector':
        """
        Create steering vector from probe weights.
        
        Direction of high trust.
        """
        # Probe weights: [hidden_dim]
        weights = torch.tensor(probe.probe.coef_[0], dtype=torch.float32)
        return cls(weights, layer_name)
    
    @classmethod
    def from_activation_difference(
        cls,
        high_trust_states: torch.Tensor,
        low_trust_states: torch.Tensor,
        layer_name: str
    ) -> 'SteeringVector':
        """
        Create steering vector as difference of mean activations.
        
        Args:
            high_trust_states: [n_samples, hidden_dim]
            low_trust_states: [n_samples, hidden_dim]
        """
        high_mean = high_trust_states.mean(dim=0)
        low_mean = low_trust_states.mean(dim=0)
        difference = high_mean - low_mean
        return cls(difference, layer_name)


class ActivationSteerer:
    """Performs activation steering during generation."""
    
    def __init__(self, model: HookedModel):
        self.model = model
        self.steering_vectors = {}
        self.steering_strength = 1.0
    
    def register_steering_vector(self, vector: SteeringVector):
        """Register a steering vector for a layer."""
        self.steering_vectors[vector.layer_name] = vector
    
    def set_steering_strength(self, strength: float):
        """Set strength of steering (multiplier)."""
        self.steering_strength = strength
    
    def _steering_hook(self, layer_name: str):
        """Create a hook that adds steering vector."""
        vector = self.steering_vectors[layer_name]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Convert steering vector to match hidden states dtype and device
            steering_vec = vector.vector.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            # Add steering vector
            steered = hidden_states + self.steering_strength * steering_vec
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered
        
        return hook_fn
    
    def run_with_steering(
        self,
        conversation: Conversation,
        layer_name: str,
        steering_strength: float = 1.0
    ) -> ModelOutput:
        """
        Run conversation with activation steering.
        
        Args:
            conversation: Conversation to run
            layer_name: Layer to apply steering at
            steering_strength: Strength of steering
        """
        if layer_name not in self.steering_vectors:
            raise ValueError(f"No steering vector registered for {layer_name}")
        
        # Set strength
        old_strength = self.steering_strength
        self.steering_strength = steering_strength
        
        # Register hook
        layer = dict(self.model.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(self._steering_hook(layer_name))
        
        try:
            # Run conversation
            output = self.model.run_conversation(conversation)
        finally:
            # Remove hook
            handle.remove()
            self.steering_strength = old_strength
        
        return output
    
    def run_steering_experiment(
        self,
        conversations: List[Conversation],
        layer_name: str,
        steering_strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ) -> Dict[float, List[ModelOutput]]:
        """
        Run steering experiment with multiple strengths.
        
        Returns:
            Dictionary mapping steering_strength to list of outputs
        """
        results = {}
        
        for strength in steering_strengths:
            outputs = []
            for conv in conversations:
                output = self.run_with_steering(conv, layer_name, strength)
                outputs.append(output)
            results[strength] = outputs
        
        return results


def create_steering_vector_from_probe(
    probe: TrustProbe,
    layer_name: str
) -> SteeringVector:
    """Helper to create steering vector from trained probe."""
    return SteeringVector.from_probe(probe, layer_name)


def create_steering_vector_from_activations(
    high_trust_outputs: List[ModelOutput],
    low_trust_outputs: List[ModelOutput],
    layer_name: str
) -> SteeringVector:
    """Helper to create steering vector from activation differences."""
    # Extract activations
    high_states = torch.stack([
        output.hidden_states[layer_name] for output in high_trust_outputs
    ])
    low_states = torch.stack([
        output.hidden_states[layer_name] for output in low_trust_outputs
    ])
    
    return SteeringVector.from_activation_difference(
        high_states, low_states, layer_name
    )


if __name__ == "__main__":
    print("Interventions module loaded.")
    print("Use ActivationSteerer for steering experiments.")
    print("Use ActivationPatcher for patching experiments (requires custom forward pass).")

