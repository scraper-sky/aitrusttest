"""
Main experiment orchestration script.

Runs the full pipeline:
1. Generate datasets
2. Run behavioral experiments
3. Train probes
4. Run interventions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset import DatasetGenerator, Conversation
from model_runner import HookedModel, get_default_layers
from metrics import MetricsCalculator, ConversationMetrics
from controls import run_control_experiments
from probes import ProbeTrainer
from interventions import ActivationSteerer, create_steering_vector_from_probe


class ExperimentRunner:
    """Orchestrates the full experiment pipeline."""
    
    def __init__(
        self,
        model_name: str,
        data_dir: str = "data",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.generator = DatasetGenerator(seed=42)
        self.model = HookedModel(model_name, device=device)
        self.metrics_calc = MetricsCalculator()
        self.probe_trainer = ProbeTrainer()
        
        # Get layer names for hooking
        self.layer_names = get_default_layers(model_name, n_layers=5)
        print(f"Will hook layers: {self.layer_names}")
    
    def stage_generate(
        self,
        n_items: int = 50,
        n_history_turns: int = 4,
        domains: List[str] = None,
        save_path: str = None
    ) -> List[Conversation]:
        """Stage 1: Generate conversation datasets."""
        print("=" * 60)
        print("STAGE 1: Generating datasets")
        print("=" * 60)
        
        if domains is None:
            domains = ["math"]
        
        conversations = self.generator.generate_paired_dataset(
            n_base_items=n_items,
            n_history_turns=n_history_turns,
            domains=domains
        )
        
        if save_path is None:
            save_path = self.data_dir / "generated" / "conversations.json"
        
        self.generator.save_conversations(conversations, str(save_path))
        
        print(f"Generated {len(conversations)} conversations")
        print(f"  High-trust: {sum(1 for c in conversations if c.condition == 'high_trust')}")
        print(f"  Low-trust: {sum(1 for c in conversations if c.condition == 'low_trust')}")
        print(f"Saved to: {save_path}")
        
        return conversations
    
    def stage_behavioral(
        self,
        conversations: List[Conversation] = None,
        load_path: str = None,
        save_results: bool = True
    ) -> tuple[List[ConversationMetrics], pd.DataFrame]:
        """Stage 2: Run behavioral experiments and compute metrics."""
        print("=" * 60)
        print("STAGE 2: Running behavioral experiments")
        print("=" * 60)
        
        # Load conversations if needed
        if conversations is None:
            if load_path is None:
                load_path = self.data_dir / "generated" / "conversations.json"
            conversations = self.generator.load_conversations(str(load_path))
        
        print(f"Running {len(conversations)} conversations through model...")
        
        # Run model
        model_outputs = self.model.run_batch(
            conversations,
            layer_names=self.layer_names,
            extract_hidden_at="last_user_token",
            max_new_tokens=100,
            temperature=0.0
        )
        
        # Compute metrics
        print("Computing metrics...")
        metrics_list = [
            self.metrics_calc.compute_metrics(conv, output)
            for conv, output in zip(conversations, model_outputs)
        ]
        
        # Summary statistics
        stats = self.metrics_calc.compute_summary_stats(metrics_list)
        
        print("\n" + "=" * 60)
        print("BEHAVIORAL RESULTS")
        print("=" * 60)
        print(f"Overall Update Rate: {stats['overall_ur']:.3f}")
        print(f"High-trust UR: {stats['high_trust_ur']:.3f} (n={stats['high_trust_n']})")
        print(f"Low-trust UR: {stats['low_trust_ur']:.3f} (n={stats['low_trust_n']})")
        print(f"Difference: {stats['high_trust_ur'] - stats['low_trust_ur']:.3f}")
        
        if 'high_trust_final_True_ur' in stats:
            print(f"\nBy final correction truth:")
            print(f"  High-trust, True: {stats['high_trust_final_True_ur']:.3f}")
            print(f"  High-trust, False: {stats['high_trust_final_False_ur']:.3f}")
            print(f"  Low-trust, True: {stats['low_trust_final_True_ur']:.3f}")
            print(f"  Low-trust, False: {stats['low_trust_final_False_ur']:.3f}")
        
        # Convert to DataFrame
        df = self.metrics_calc.metrics_to_dataframe(metrics_list)
        
        # Save results
        if save_results:
            results_path = self.data_dir / "results" / "behavioral_results.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path, index=False)
            
            stats_path = self.data_dir / "results" / "behavioral_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\nResults saved to: {results_path}")
            print(f"Stats saved to: {stats_path}")
        
        # Create plots
        self._plot_behavioral_results(df, stats)
        
        return metrics_list, df
    
    def _plot_behavioral_results(self, df: pd.DataFrame, stats: Dict):
        """Create visualization plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Update rate by condition
        sns.barplot(data=df, x="condition", y="update_rate", ax=axes[0])
        axes[0].set_title("Update Rate by Trust Condition")
        axes[0].set_ylabel("Update Rate")
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Update rate by condition and final truth
        if "final_correction_true" in df.columns:
            sns.barplot(
                data=df,
                x="condition",
                y="update_rate",
                hue="final_correction_true",
                ax=axes[1]
            )
            axes[1].set_title("Update Rate by Condition and Final Truth")
            axes[1].set_ylabel("Update Rate")
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = self.data_dir / "results" / "behavioral_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plots saved to: {plot_path}")
        plt.close()
    
    def stage_probe(
        self,
        conversations: List[Conversation] = None,
        model_outputs: List = None,
        load_path: str = None
    ) -> Dict:
        """Stage 3: Train probes on hidden states."""
        print("=" * 60)
        print("STAGE 3: Training probes")
        print("=" * 60)
        
        # Load or use provided data
        if conversations is None or model_outputs is None:
            if load_path is None:
                load_path = self.data_dir / "generated" / "conversations.json"
            conversations = self.generator.load_conversations(str(load_path))
            
            # Re-run model if needed
            print("Running model to extract hidden states...")
            model_outputs = self.model.run_batch(
                conversations,
                layer_names=self.layer_names,
                extract_hidden_at="last_user_token"
            )
        
        # Train probes
        print("Training probes...")
        probe_results = self.probe_trainer.train_probes(
            model_outputs,
            conversations,
            self.layer_names
        )
        
        print("\n" + "=" * 60)
        print("PROBE RESULTS")
        print("=" * 60)
        for layer, metrics in probe_results.items():
            if "error" not in metrics:
                print(f"{layer}:")
                print(f"  Train Accuracy: {metrics['train_accuracy']:.3f}")
                print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
                print(f"  Test AUC: {metrics['test_auc']:.3f}")
        
        # Check correlation with behavior
        print("\nComputing correlation with update behavior...")
        metrics_list = [
            self.metrics_calc.compute_metrics(conv, output)
            for conv, output in zip(conversations, model_outputs)
        ]
        
        correlations = {}
        for layer_name in self.layer_names:
            if layer_name in self.probe_trainer.probes:
                try:
                    corr = self.probe_trainer.correlate_with_behavior(
                        model_outputs, metrics_list, layer_name
                    )
                    correlations[layer_name] = corr
                    print(f"\n{layer_name}:")
                    print(f"  Probe-behavior correlation: {corr['overall_correlation']:.3f}")
                    print(f"  High-trust mean score: {corr['high_trust_mean_score']:.3f}")
                    print(f"  Low-trust mean score: {corr['low_trust_mean_score']:.3f}")
                except Exception as e:
                    print(f"Error computing correlation for {layer_name}: {e}")
        
        # Save results
        results_path = self.data_dir / "results" / "probe_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "probe_metrics": probe_results,
                "correlations": correlations
            }, f, indent=2, default=str)
        
        print(f"\nProbe results saved to: {results_path}")
        
        return probe_results
    
    def stage_intervention(
        self,
        conversations: List[Conversation] = None,
        layer_name: str = None,
        steering_strengths: List[float] = None
    ) -> Dict:
        """Stage 4: Run intervention experiments."""
        print("=" * 60)
        print("STAGE 4: Running interventions")
        print("=" * 60)
        
        if conversations is None:
            load_path = self.data_dir / "generated" / "conversations.json"
            conversations = self.generator.load_conversations(str(load_path))
        
        # Select a subset for intervention (faster)
        test_conversations = conversations[:20]
        
        # Get best layer from probes
        if layer_name is None:
            # Use middle layer as default
            layer_name = self.layer_names[len(self.layer_names) // 2]
        
        # Create steering vector from probe
        if layer_name in self.probe_trainer.probes:
            probe = self.probe_trainer.probes[layer_name]
            from interventions import SteeringVector
            steering_vector = SteeringVector.from_probe(probe, layer_name)
        else:
            print(f"Warning: No probe for {layer_name}, creating from activations...")
            # Would need to extract activations first
            raise ValueError("Need trained probe or activation data")
        
        # Run steering experiment
        steerer = ActivationSteerer(self.model)
        steerer.register_steering_vector(steering_vector)
        
        if steering_strengths is None:
            steering_strengths = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        print(f"Running steering experiment with strengths: {steering_strengths}")
        steering_results = steerer.run_steering_experiment(
            test_conversations,
            layer_name,
            steering_strengths
        )
        
        # Compute metrics for each strength
        intervention_metrics = {}
        for strength, outputs in steering_results.items():
            metrics = [
                self.metrics_calc.compute_metrics(conv, output)
                for conv, output in zip(test_conversations, outputs)
            ]
            avg_ur = sum(m.update_rate for m in metrics) / len(metrics)
            intervention_metrics[strength] = {
                "mean_update_rate": avg_ur,
                "n": len(metrics)
            }
        
        print("\n" + "=" * 60)
        print("INTERVENTION RESULTS")
        print("=" * 60)
        for strength, metrics in sorted(intervention_metrics.items()):
            print(f"Steering strength {strength:+.1f}: UR = {metrics['mean_update_rate']:.3f}")
        
        # Save results
        results_path = self.data_dir / "results" / "intervention_results.json"
        with open(results_path, 'w') as f:
            json.dump(intervention_metrics, f, indent=2)
        
        print(f"\nIntervention results saved to: {results_path}")
        
        return intervention_metrics


def main():
    parser = argparse.ArgumentParser(description="Run earned trust experiments")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["generate", "behavioral", "probe", "intervention", "all"],
        help="Which stage to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (HuggingFace identifier)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=50,
        help="Number of base items to generate"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(
        model_name=args.model,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Run stages
    if args.stage == "generate" or args.stage == "all":
        runner.stage_generate(n_items=args.n_items)
    
    if args.stage == "behavioral" or args.stage == "all":
        metrics, df = runner.stage_behavioral()
    
    if args.stage == "probe" or args.stage == "all":
        runner.stage_probe()
    
    if args.stage == "intervention" or args.stage == "all":
        runner.stage_intervention()


if __name__ == "__main__":
    main()

