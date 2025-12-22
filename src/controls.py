"""
Control experiments to rule out confounds.

Implements:
- Shuffled-text control: Same structure, different correctness
- Authority-cue control: Status cues vs track record
- Memory-length control: Varying history length
"""

import random
from typing import List, Dict
from dataset import Conversation, DatasetGenerator


class ControlExperiment:
    """Base class for control experiments."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)


class ShuffledTextControl(ControlExperiment):
    """
    Shuffled-text control: Keep style/length identical but swap correctness.
    
    This tests whether the effect tracks correctness history rather than
    superficial text patterns.
    """
    
    def create_control_conversations(
        self,
        base_conversations: List[Conversation]
    ) -> List[Conversation]:
        """
        Create control conversations by shuffling which corrections are correct.
        
        For each base conversation, create a version where:
        - Same structure and style
        - Same number of corrections
        - But correctness labels are shuffled/randomized
        """
        control_conversations = []
        
        for base_conv in base_conversations:
            # Create a shuffled version
            # We'll keep the same structure but randomize correctness
            shuffled_correctness = base_conv.history_correctness.copy()
            self.rng.shuffle(shuffled_correctness)
            
            # Create new conversation with shuffled correctness
            # This is a simplified version - in practice, you'd need to
            # swap the actual correction values in the turns
            control_conv = Conversation(
                condition=f"{base_conv.condition}_shuffled",
                turns=base_conv.turns.copy(),  # Would need to modify these
                history_correctness=shuffled_correctness,
                final_correction_true=base_conv.final_correction_true,
                item_id=f"{base_conv.item_id}_shuffled",
                domain=base_conv.domain
            )
            
            control_conversations.append(control_conv)
        
        return control_conversations


class AuthorityCueControl(ControlExperiment):
    """
    Authority-cue control: Add status cues independent of track record.
    
    Tests whether explicit authority cues (PhD, confident tone) are
    weaker than earned track record.
    """
    
    def add_authority_cues(
        self,
        conversation: Conversation,
        cue_type: str = "phd"
    ) -> Conversation:
        """
        Add authority cues to conversation turns.
        
        Args:
            conversation: Base conversation
            cue_type: "phd", "expert", "confident_tone"
        """
        new_turns = []
        
        for turn in conversation.turns:
            if turn["role"] == "user":
                # Add authority cue to user messages
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
        """Create dataset with authority cues added."""
        return [
            self.add_authority_cues(conv, cue_type)
            for conv in base_conversations
        ]


class MemoryLengthControl(ControlExperiment):
    """
    Memory-length control: Vary history length to test trust accumulation.
    
    Tests whether longer histories lead to stronger trust effects.
    """
    
    def create_variable_length_conversations(
        self,
        base_item: dict,
        generator: DatasetGenerator,
        lengths: List[int] = [1, 2, 3, 4, 5]
    ) -> List[Conversation]:
        """
        Create conversations with varying history lengths.
        
        Args:
            base_item: Base item to use
            generator: DatasetGenerator instance
            lengths: List of history lengths to test
        """
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
    """
    Run specified control experiments.
    
    Args:
        base_conversations: Base conversations to create controls from
        which_controls: List of controls to run:
            - "shuffled_text"
            - "authority_cue"
            - "memory_length"
    
    Returns:
        Dictionary mapping control name to list of conversations
    """
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
        # This one needs base items, not conversations
        # Would need to be handled separately
        pass
    
    return results


if __name__ == "__main__":
    # Example usage
    from dataset import DatasetGenerator
    
    generator = DatasetGenerator()
    base_convs = generator.generate_paired_dataset(n_base_items=10, n_history_turns=4)
    
    # Run shuffled text control
    control = ShuffledTextControl()
    shuffled = control.create_control_conversations(base_convs[:5])
    
    print(f"Created {len(shuffled)} shuffled control conversations")
    
    # Run authority cue control
    authority_control = AuthorityCueControl()
    authority_convs = authority_control.create_authority_control_dataset(
        base_convs[:5],
        cue_type="phd"
    )
    
    print(f"Created {len(authority_convs)} authority-cue conversations")
    print("\nExample with authority cue:")
    for turn in authority_convs[0].turns[:3]:
        print(f"{turn['role']}: {turn['content']}")

