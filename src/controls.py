import random
from typing import List, Dict
from dataset import Conversation, DatasetGenerator


class ControlExperiment:
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)


class ShuffledTextControl(ControlExperiment):
    
    def create_control_conversations(
        self,
        base_conversations: List[Conversation]
    ) -> List[Conversation]:
        control_conversations = []
        
        for base_conv in base_conversations:
            shuffled_correctness = base_conv.history_correctness.copy()
            self.rng.shuffle(shuffled_correctness)
            
            control_conv = Conversation(
                condition=f"{base_conv.condition}_shuffled",
                turns=base_conv.turns.copy(),
                history_correctness=shuffled_correctness,
                final_correction_true=base_conv.final_correction_true,
                item_id=f"{base_conv.item_id}_shuffled",
                domain=base_conv.domain
            )
            
            control_conversations.append(control_conv)
        
        return control_conversations


class AuthorityCueControl(ControlExperiment):
    
    def add_authority_cues(
        self,
        conversation: Conversation,
        cue_type: str = "phd"
    ) -> Conversation:
        new_turns = []
        
        for turn in conversation.turns:
            if turn["role"] == "user":
                if cue_type == "phd":
                    prefix = "As someone with a PhD in mathematics, "
                elif cue_type == "expert":
                    prefix = "As an expert in this field, "
                elif cue_type == "confident_tone":
                    prefix = "I'm absolutely certain that "
                else:
                    prefix = ""
                
                new_content = prefix + turn["content"]
                new_turns.append({**turn, "content": new_content})
            else:
                new_turns.append(turn)
        
        return Conversation(
            condition=f"{conversation.condition}_authority_{cue_type}",
            turns=new_turns,
            history_correctness=conversation.history_correctness,
            final_correction_true=conversation.final_correction_true,
            item_id=f"{conversation.item_id}_authority",
            domain=conversation.domain
        )
    
    def create_authority_control_dataset(
        self,
        base_conversations: List[Conversation],
        cue_type: str = "phd"
    ) -> List[Conversation]:
        return [
            self.add_authority_cues(conv, cue_type)
            for conv in base_conversations
        ]


class MemoryLengthControl(ControlExperiment):
    
    def create_variable_length_conversations(
        self,
        base_item: dict,
        generator: DatasetGenerator,
        lengths: List[int] = [1, 2, 3, 4, 5]
    ) -> List[Conversation]:
        conversations = []
        
        for length in lengths:
            for condition in ["high_trust", "low_trust"]:
                conv = generator.create_conversation(
                    base_item,
                    condition,
                    n_history_turns=length,
                    final_correction_true=True
                )
                conversations.append(conv)
        
        return conversations


def run_control_experiments(
    base_conversations: List[Conversation],
    which_controls: List[str] = ["shuffled_text"]
) -> Dict[str, List[Conversation]]:
    results = {}
    
    if "shuffled_text" in which_controls:
        control = ShuffledTextControl()
        results["shuffled_text"] = control.create_control_conversations(base_conversations)
    
    if "authority_cue" in which_controls:
        control = AuthorityCueControl()
        results["authority_cue"] = control.create_authority_control_dataset(
            base_conversations,
            cue_type="phd"
        )
    
    if "memory_length" in which_controls:
        pass
    
    return results


if __name__ == "__main__":
    from dataset import DatasetGenerator
    
    generator = DatasetGenerator()
    base_convs = generator.generate_paired_dataset(n_base_items=10, n_history_turns=4)
    
    control = ShuffledTextControl()
    shuffled = control.create_control_conversations(base_convs[:5])
    
    print(f"Created {len(shuffled)} shuffled control conversations")
    
    authority_control = AuthorityCueControl()
    authority_convs = authority_control.create_authority_control_dataset(
        base_convs[:5],
        cue_type="phd"
    )
    
    print(f"Created {len(authority_convs)} authority-cue conversations")
    print("\nExample with authority cue:")
    for turn in authority_convs[0].turns[:3]:
        print(f"{turn['role']}: {turn['content']}")
