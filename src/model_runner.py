import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from dataset import Conversation


@dataclass
class ModelOutput:
    conversation: Conversation
    final_response: str
    hidden_states: Dict[str, torch.Tensor]
    hook_positions: Dict[str, int]
    full_response_tokens: List[str]


class HookedModel:
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_half_precision: bool = True
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        if device == "cuda" and use_half_precision:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
                self.model = model
            except Exception as e:
                print(f"Warning: device_map='auto' failed, trying manual device placement: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16
                )
                self.model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if use_half_precision else torch.float32
            )
            self.model = model.to(device)
        
        self.model.eval()
        
        self.activations = {}
        self.hook_handles = []
    
    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook_fn
    
    def register_hooks(self, layer_names: List[str]):
        self.clear_hooks()
        self.activations = {}
        
        model_to_hook = self.model
        if hasattr(self.model, 'model'):
            model_to_hook = self.model.model
        elif hasattr(self.model, 'transformer'):
            model_to_hook = self.model.transformer
        
        all_module_names = list(dict(model_to_hook.named_modules()).keys())
        
        for layer_name in layer_names:
            try:
                layer = dict(model_to_hook.named_modules())[layer_name]
                handle = layer.register_forward_hook(self._make_hook(layer_name))
                self.hook_handles.append(handle)
            except KeyError:
                print(f"Warning: Layer {layer_name} not found, skipping")
                layer_num = layer_name.split('.')[-1]
                possible_matches = [name for name in all_module_names if f".{layer_num}" in name or f".{layer_num}." in name]
                if possible_matches:
                    print(f"  Possible matches: {possible_matches[:3]}")
                else:
                    layer_keywords = ["layer", "transformer", "gpt_neox", "h."]
                    relevant_names = [name for name in all_module_names if any(kw in name.lower() for kw in layer_keywords)]
                    if relevant_names:
                        print(f"  Sample layer names in model: {relevant_names[:5]}")
    
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def format_conversation(self, conversation: Conversation) -> str:
        parts = []
        for turn in conversation.turns:
            if turn["role"] == "user":
                parts.append(f"User: {turn['content']}")
            else:
                parts.append(f"Assistant: {turn['content']}")
        
        return "\n".join(parts)
    
    def run_conversation(
        self,
        conversation: Conversation,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        extract_hidden_at: Optional[str] = None
    ) -> ModelOutput:
        prompt = self.format_conversation(conversation)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        hook_positions = {}
        
        hidden_states = {}
        if extract_hidden_at == "last_user_token":
            with torch.no_grad():
                self.activations = {}
                model_outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
            
            if self.activations:
                for layer_name, activations in self.activations.items():
                    seq_len = activations.shape[1]
                    idx = min(input_ids.shape[1] - 1, seq_len - 1)
                    hidden_states[layer_name] = activations[0, idx, :]
                    hook_positions[layer_name] = idx
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
        
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        
        if extract_hidden_at != "last_user_token":
            self.activations = {}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **generation_kwargs
            )
        
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(
            generated_ids[input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        if extract_hidden_at == "first_assistant_token" and self.activations:
            for layer_name, activations in self.activations.items():
                seq_len = activations.shape[1]
                idx = min(input_ids.shape[1], seq_len - 1)
                hidden_states[layer_name] = activations[0, idx, :]
                hook_positions[layer_name] = idx
        
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
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    
    if hasattr(config, 'num_hidden_layers'):
        total_layers = config.num_hidden_layers
        layer_indices = [int(i * total_layers / (n_layers + 1)) for i in range(1, n_layers + 1)]
        
        model_name_lower = model_name.lower()
        if "gpt2" in model_name_lower:
            pattern = "transformer.h.{}"
        elif "pythia" in model_name_lower or "gpt-neox" in model_name_lower or "gpt_neox" in model_name_lower:
            pattern = "gpt_neox.layers.{}"
        elif any(x in model_name_lower for x in ["gemma", "llama", "mistral", "phi"]):
            pattern = "model.layers.{}"
        else:
            pattern = "model.layers.{}"
        
        return [pattern.format(idx) for idx in layer_indices]
    
    return [f"transformer.h.{idx}" for idx in [2, 4, 6, 8, 10]]


def list_available_layers(model):
    """Helper function to list available layer names in a model."""
    all_modules = dict(model.named_modules())
    layer_names = [name for name in all_modules.keys() if 'layer' in name.lower() or 'transformer.h' in name or 'gpt_neox' in name]
    return sorted(layer_names)


if __name__ == "__main__":
    model_name = "gpt2"
    
    print(f"Loading model: {model_name}")
    hooked_model = HookedModel(model_name, device="cpu")
    
    from dataset import DatasetGenerator
    generator = DatasetGenerator()
    test_item = generator.generate_math_items(1)[0]
    conv = generator.create_conversation(test_item, "high_trust", n_history_turns=3)
    
    layers = get_default_layers(model_name, n_layers=3)
    print(f"Hooking layers: {layers}")
    
    output = hooked_model.run_batch(
        [conv],
        layer_names=layers,
        extract_hidden_at="last_user_token"
    )[0]
    
    print(f"\nFinal response: {output.final_response}")
    print(f"Extracted hidden states from {len(output.hidden_states)} layers")
