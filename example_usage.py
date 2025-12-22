"""
Example usage script for the earned trust experiment.

This demonstrates how to use the components individually.
"""

from src import (
    DatasetGenerator,
    HookedModel,
    MetricsCalculator,
    ProbeTrainer,
    ActivationSteerer
)

def example_basic_usage():
    """Basic example: generate data and run behavioral experiment."""
    
    # 1. Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator(seed=42)
    conversations = generator.generate_paired_dataset(
        n_base_items=20,  # Small for testing
        n_history_turns=3,
        domains=["math"]
    )
    
    # Save it
    generator.save_conversations(conversations, "data/generated/example_conversations.json")
    print(f"Generated {len(conversations)} conversations")
    
    # 2. Load model (use a small model for testing)
    print("\nLoading model...")
    model = HookedModel("gpt2", device="cpu")  # Use CPU for small models
    
    # 3. Run conversations
    print("Running conversations through model...")
    from src.model_runner import get_default_layers
    layer_names = get_default_layers("gpt2", n_layers=3)
    
    model_outputs = model.run_batch(
        conversations[:10],  # Just first 10 for speed
        layer_names=layer_names,
        extract_hidden_at="last_user_token",
        max_new_tokens=50
    )
    
    # 4. Compute metrics
    print("Computing metrics...")
    metrics_calc = MetricsCalculator()
    metrics_list = [
        metrics_calc.compute_metrics(conv, output)
        for conv, output in zip(conversations[:10], model_outputs)
    ]
    
    # 5. Summary stats
    stats = metrics_calc.compute_summary_stats(metrics_list)
    print("\nResults:")
    print(f"High-trust UR: {stats.get('high_trust_ur', 0):.3f}")
    print(f"Low-trust UR: {stats.get('low_trust_ur', 0):.3f}")
    
    return conversations, model_outputs, metrics_list


def example_probe_training(conversations, model_outputs):
    """Example: train probes on hidden states."""
    
    print("\n" + "="*60)
    print("Training probes...")
    
    probe_trainer = ProbeTrainer()
    
    # Train probes
    results = probe_trainer.train_probes(
        model_outputs,
        conversations,
        layer_names=["model.layers.5", "model.layers.10"],  # Example layers
        train_split=0.8
    )
    
    print("Probe results:")
    for layer, metrics in results.items():
        if "error" not in metrics:
            print(f"  {layer}: Test AUC = {metrics['test_auc']:.3f}")
    
    return probe_trainer


def example_steering(conversations, model_outputs, probe_trainer):
    """Example: run steering intervention."""
    
    print("\n" + "="*60)
    print("Running steering intervention...")
    
    # Get a probe
    layer_name = list(probe_trainer.probes.keys())[0]
    probe = probe_trainer.probes[layer_name]
    
    # Create steering vector
    from src.interventions import SteeringVector
    steering_vector = SteeringVector.from_probe(probe, layer_name)
    
    # Create steerer
    model = HookedModel("gpt2", device="cpu")
    steerer = ActivationSteerer(model)
    steerer.register_steering_vector(steering_vector)
    
    # Run with different strengths
    test_conv = conversations[0]
    for strength in [-1.0, 0.0, 1.0]:
        output = steerer.run_with_steering(test_conv, layer_name, strength)
        print(f"Strength {strength:+.1f}: {output.final_response[:100]}...")


if __name__ == "__main__":
    # Run basic example
    conversations, model_outputs, metrics = example_basic_usage()
    
    # Uncomment to run probe training (requires more data)
    # probe_trainer = example_probe_training(conversations, model_outputs)
    
    # Uncomment to run steering (requires trained probes)
    # example_steering(conversations, model_outputs, probe_trainer)

