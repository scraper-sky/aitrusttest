# Earned Trust: Latent User-Model Variable Gating Epistemic Updating

## Project Overview

This project tests whether language models maintain an implicit, evolving "user reliability/trust" state in multi-turn dialogue that causally controls whether the model accepts or rejects later user corrections.

## Hypothesis

- **H1**: After a user demonstrates a track record of being right vs wrong, the model encodes a latent "user reliability/trust" variable.
- **H2**: That variable gates epistemic updating—higher trust increases probability of updating toward user's claim.
- **H3**: Intervening on that latent state (patching/steering) can shift update behavior.

## Project Structure

```
.
├── data/
│   ├── generated/          # Generated conversation datasets
│   └── results/            # Experimental results
├── src/
│   ├── dataset.py          # Conversation dataset generator
│   ├── model_runner.py     # Model inference with hook extraction
│   ├── metrics.py          # Update rate, confidence, verification metrics
│   ├── controls.py         # Control experiments (shuffled-text, etc.)
│   ├── probes.py           # Linear probe training for trust detection
│   ├── interventions.py    # Activation patching and steering
│   └── experiment.py       # Main experiment orchestration
├── notebooks/              # Analysis notebooks
├── requirements.txt
└── README.md
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Generate datasets: `python src/experiment.py --stage generate`
3. Run behavioral experiments: `python src/experiment.py --stage behavioral`
4. Train probes: `python src/experiment.py --stage probe`
5. Run interventions: `python src/experiment.py --stage intervention`

## Experimental Design

Each trial consists of:
- **Turn 0**: Assistant answers a simple question
- **Turns 1-N**: User provides corrections (right/wrong depending on condition)
- **Final turn**: User asserts correction; measure acceptance

Two conditions:
- **High-trust**: User makes 3-5 correct corrections
- **Low-trust**: User makes 3-5 incorrect corrections

## Metrics

- **Update Rate (UR)**: Fraction of trials where assistant's final stance matches user's correction
- **Confidence**: Classification of response confidence level
- **Verification**: Whether model requests verification/sources

