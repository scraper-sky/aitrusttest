"""
Model runner with hidden state extraction hooks.

Runs conversations through a model and extracts:
- Final responses
- Hidden states at key positions
- Attention patterns (optional)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from dataset import Conversation


@dataclass
class ModelOutput:
    """Output from model run including response and hidden states."""
    conversation: Conversation
    final_response: str
    hidden_states: Dict[str, torch.Tensor]  # layer_name -> tensor
    hook_positions: Dict[str, int]  # position name -> token index
    full_response_tokens: List[str]


class HookedModel:
    """Wrapper around a model with hook extraction capabilities."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_half_precision: bool = True
    ):
        """
        Initialize model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: "auto", "cuda", or "cpu"
            use_half_precision: Use float16 for faster inference
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # For larger models, use device_map for automatic GPU allocation
        # For smaller models or CPU, use .to(device)
        if device == "cuda" and use_half_precision:
            # Use device_map for larger models (better memory management)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"  # Automatically distributes across available GPUs
                )
                self.model = model
            except Exception as e:
                print(f"Warning: device_map='auto' failed, trying manual device placement: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
                self.model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if use_half_precision else torch.float32
            )
            self.model = model.to(device)
        
        self.model.eval()
        
        # Storage for hooks
        self.activations = {}
        self.hook_handles = []
    
    def _make_hook(self, name: str):
        """Create a hook function to store activations."""
        def hook_fn(module, input, output):
            # output is typically a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook_fn
    
    def register_hooks(self, layer_names: List[str]):
        """
        Register hooks at specified layers.
        
        Args:
            layer_names: List of layer names like ["model.layers.10", "model.layers.20"]
        """
        self.clear_hooks()
        self.activations = {}
        
        for layer_name in layer_names:
            try:
                layer = dict(self.model.named_modules())[layer_name]
                handle = layer.register_forward_hook(self._make_hook(layer_name))
                self.hook_handles.append(handle)
            except KeyError:
                print(f"Warning: Layer {layer_name} not found, skipping")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def format_conversation(self, conversation: Conversation) -> str:
        """Format conversation as a single prompt string."""
        # Simple format: concatenate turns
        parts = []
        for turn in conversation.turns:
            if turn["role"] == "user":
                parts.append(f"User: {turn['content']}")
            else:
                parts.append(f"Assistant: {turn['content']}")
        
        # Add final user turn without assistant response yet
        return "\n".join(parts)
    
    def run_conversation(
        self,
        conversation: Conversation,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        extract_hidden_at: Optional[str] = None
    ) -> ModelOutput:
        """
        Run a conversation through the model.
        
        Args:
            conversation: Conversation to run
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            extract_hidden_at: Where to extract hidden states:
                - "last_user_token": Last token of final user message
                - "first_assistant_token": First token of assistant response
                - None: Don't extract
        """
        # Format prompt (everything except final assistant response)
        prompt = self.format_conversation(conversation)
        
        # Tokenize with attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Initialize hook_positions dict for tracking
        hook_positions = {}
        
        # Extract hidden states BEFORE generation (if needed)
        hidden_states = {}
        if extract_hidden_at == "last_user_token":
            # Do a forward pass to get hidden states at the last user token
            with torch.no_grad():
                # Clear any previous activations
                self.activations = {}
                # Forward pass on input only
                model_outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
            
            # Extract from hooks (captured during forward pass)
            if self.activations:
                for layer_name, activations in self.activations.items():
                    # activations shape: [batch, seq_len, hidden_dim]
                    seq_len = activations.shape[1]
                    # Use last token of input
                    idx = min(input_ids.shape[1] - 1, seq_len - 1)
                    hidden_states[layer_name] = activations[0, idx, :]
                    hook_positions[layer_name] = idx
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        
        # Only add temperature/do_sample if temperature > 0
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
        
        # Add attention mask if available
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        
        # Generate response (clear hooks first to avoid capturing generation activations)
        if extract_hidden_at != "last_user_token":
            self.activations = {}  # Clear for generation if we're not extracting from input
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **generation_kwargs
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(
            generated_ids[input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # If extracting from first assistant token, get it from generation hooks
        if extract_hidden_at == "first_assistant_token" and self.activations:
            for layer_name, activations in self.activations.items():
                seq_len = activations.shape[1]
                # Use position right after input (first generated token)
                idx = min(input_ids.shape[1], seq_len - 1)
                hidden_states[layer_name] = activations[0, idx, :]
                hook_positions[layer_name] = idx
        
        # Tokenize full response for analysis
        full_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        
        return ModelOutput(
            conversation=conversation,
            final_response=generated_text,
            hidden_states=hidden_states,
            hook_positions=hook_positions,
            full_response_tokens=full_tokens
        )
    
    def run_batch(
        self,
        conversations: List[Conversation],
        layer_names: List[str],
        extract_hidden_at: str = "last_user_token",
        **kwargs
    ) -> List[ModelOutput]:
        """Run a batch of conversations."""
        # Register hooks
        self.register_hooks(layer_names)
        
        results = []
        for conv in conversations:
            output = self.run_conversation(
                conv,
                extract_hidden_at=extract_hidden_at,
                **kwargs
            )
            results.append(output)
        
        return results


def get_default_layers(model_name: str, n_layers: int = 5) -> List[str]:
    """
    Get default layer names for hooking.
    
    Tries to extract layers evenly spaced through the model.
    """
    # Try to load model config to get layer names
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    
    # Common patterns
    if hasattr(config, 'num_hidden_layers'):
        total_layers = config.num_hidden_layers
        layer_indices = [int(i * total_layers / (n_layers + 1)) for i in range(1, n_layers + 1)]
        
        # Try to detect the correct pattern by loading a small part of the model
        # Common patterns for different architectures:
        # GPT-2: transformer.h.{i}
        # LLaMA/Mistral/Gemma: model.layers.{i}
        # GPT-NeoX: gpt_neox.layers.{i}
        
        # Detect architecture from model name
        model_name_lower = model_name.lower()
        if "gpt2" in model_name_lower:
            pattern = "transformer.h.{}"
        elif any(x in model_name_lower for x in ["gemma", "llama", "mistral", "phi"]):
            pattern = "model.layers.{}"
        else:
            # Default: try model.layers first (most common for modern models)
            pattern = "model.layers.{}"
        
        # Return layer names
        return [pattern.format(idx) for idx in layer_indices]
    
    # Fallback: return some common layer names
    return [f"transformer.h.{idx}" for idx in [2, 4, 6, 8, 10]]


if __name__ == "__main__":
    # Example usage
    # Note: Replace with a model you have access to
    model_name = "gpt2"  # Small model for testing
    
    print(f"Loading model: {model_name}")
    hooked_model = HookedModel(model_name, device="cpu")
    
    # Create a test conversation
    from dataset import DatasetGenerator
    generator = DatasetGenerator()
    test_item = generator.generate_math_items(1)[0]
    conv = generator.create_conversation(test_item, "high_trust", n_history_turns=3)
    
    # Get layer names
    layers = get_default_layers(model_name, n_layers=3)
    print(f"Hooking layers: {layers}")
    
    # Run
    output = hooked_model.run_batch(
        [conv],
        layer_names=layers,
        extract_hidden_at="last_user_token"
    )[0]
    
    print(f"\nFinal response: {output.final_response}")
    print(f"Extracted hidden states from {len(output.hidden_states)} layers")

