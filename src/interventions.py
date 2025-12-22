import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

from dataset import Conversation
from model_runner import HookedModel, ModelOutput
from metrics import ConversationMetrics
from probes import TrustProbe


class ActivationPatcher:
    
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
        return target_output
    
    def run_patching_experiment(
        self,
        high_trust_conv: Conversation,
        low_trust_conv: Conversation,
        layer_name: str,
        token_position: str = "last_user_token"
    ) -> Tuple[ModelOutput, ModelOutput, ModelOutput]:
        high_output = self.model.run_conversation(high_trust_conv)
        low_output = self.model.run_conversation(low_trust_conv)
        
        patched_output = self.patch_activations(
            high_output, low_output, layer_name, token_position
        )
        
        return high_output, low_output, patched_output


class SteeringVector:
    
    def __init__(self, vector: torch.Tensor, layer_name: str):
        self.vector = vector
        self.layer_name = layer_name
    
    @classmethod
    def from_probe(cls, probe: TrustProbe, layer_name: str) -> 'SteeringVector':
        weights = torch.tensor(probe.probe.coef_[0], dtype=torch.float32)
        return cls(weights, layer_name)
    
    @classmethod
    def from_activation_difference(
        cls,
        high_trust_states: torch.Tensor,
        low_trust_states: torch.Tensor,
        layer_name: str
    ) -> 'SteeringVector':
        high_mean = high_trust_states.mean(dim=0)
        low_mean = low_trust_states.mean(dim=0)
        difference = high_mean - low_mean
        return cls(difference, layer_name)


class ActivationSteerer:
    
    def __init__(self, model: HookedModel):
        self.model = model
        self.steering_vectors = {}
        self.steering_strength = 1.0
    
    def register_steering_vector(self, vector: SteeringVector):
        self.steering_vectors[vector.layer_name] = vector
    
    def set_steering_strength(self, strength: float):
        self.steering_strength = strength
    
    def _steering_hook(self, layer_name: str):
        vector = self.steering_vectors[layer_name]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            steering_vec = vector.vector.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
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
        if layer_name not in self.steering_vectors:
            raise ValueError(f"No steering vector registered for {layer_name}")
        
        old_strength = self.steering_strength
        self.steering_strength = steering_strength
        
        layer = dict(self.model.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(self._steering_hook(layer_name))
        
        try:
            output = self.model.run_conversation(conversation)
        finally:
            handle.remove()
            self.steering_strength = old_strength
        
        return output
    
    def run_steering_experiment(
        self,
        conversations: List[Conversation],
        layer_name: str,
        steering_strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ) -> Dict[float, List[ModelOutput]]:
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
    return SteeringVector.from_probe(probe, layer_name)


def create_steering_vector_from_activations(
    high_trust_outputs: List[ModelOutput],
    low_trust_outputs: List[ModelOutput],
    layer_name: str
) -> SteeringVector:
    high_states = torch.stack([
        output.hidden_states[layer_name] for output in high_trust_outputs
    ])
    low_states = torch.stack([
        output.hidden_states[layer_name] for output in low_trust_outputs
    ])
    
    return SteeringVector.from_activation_difference(
        high_states, low_states, layer_name
    )
