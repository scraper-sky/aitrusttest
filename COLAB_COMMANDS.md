# Colab Commands - Re-running Experiments with Transcript Control and Statistical Tests

## Quick Setup

```python
# Install dependencies
!pip install torch transformers numpy pandas scikit-learn matplotlib seaborn tqdm accelerate datasets scipy

# Clone repository
!git clone https://github.com/scraper-sky/aitrusttest.git
%cd aitrusttest
```

---

## Option 1: Re-run with Transcript Control (Recommended)

This uses neutral assistant responses ("Noted. Let's continue.") instead of "I see. Thank you for the correction." to reduce transcript feature leakage.

### For Phi-2 (Best Results - Has Behavioral Effect + Causal Evidence)

```python
# Pull latest changes
!cd /content/aitrusttest && git pull

# Run full experiment with transcript control
!python src/experiment.py --stage all --model microsoft/phi-2 --n-items 50 --neutral-responses
```

**What this does**:
- Uses neutral assistant responses in history
- Includes statistical tests (CIs, p-values)
- Tracks probe scores during steering
- Enhanced stratification analysis

**Expected Output**:
- Behavioral results with 95% CI and p-values
- Probe results
- Intervention results with probe scores
- Epistemic vs. deference analysis

---

### For GPT-2 (Also Shows Behavioral Effect)

```python
!cd /content/aitrusttest && git pull
!python src/experiment.py --stage all --model gpt2 --n-items 50 --neutral-responses
```

---

### For Pythia-1B (Strong Probes, No Behavior)

```python
!cd /content/aitrusttest && git pull
!python src/experiment.py --stage all --model EleutherAI/pythia-1b --n-items 50 --neutral-responses
```

**Key Question**: Will probes still get high AUC with neutral responses? If yes → stronger evidence for true trust encoding. If no → likely transcript features.

---

## Option 2: Re-run WITHOUT Transcript Control (Baseline Comparison)

To compare results:

```python
!cd /content/aitrusttest && git pull
!python src/experiment.py --stage all --model microsoft/phi-2 --n-items 50
```

(No `--neutral-responses` flag = original "I see. Thank you" responses)

---

## What You'll Get Now

### 1. Statistical Tests (Automatic)

**Behavioral Results Now Include**:
```
Difference: 0.050 [95% CI: -0.012, 0.112], p=0.1234
```

This tells you:
- Effect size: 5.0%
- Confidence interval: [-1.2%, 11.2%]
- Statistical significance: p-value

### 2. Epistemic Analysis (Enhanced)

```
By final correction truth (epistemic check):
  High-trust, True: 0.636 (n=25)
  High-trust, False: 0.786 (n=25)
  → Both conditions accept TRUE corrections more (epistemically sensible)
```

This tells you if trust is epistemically sensible or just deference.

### 3. Probe Scores Under Steering

```
Steering strength -2.0: UR = 0.950, Probe score = 0.234
Steering strength +0.0: UR = 0.775, Probe score = 0.661
```

This helps verify if steering direction is correct.

---

## Recommended Run Sequence

### Step 1: Phi-2 with Transcript Control

```python
!cd /content/aitrusttest && git pull
!python src/experiment.py --stage all --model microsoft/phi-2 --n-items 50 --neutral-responses
```

**Why First**: Phi-2 has the strongest findings (behavioral effect + causal evidence). Transcript control will test if probes still work.

### Step 2: Compare to Baseline (Optional)

```python
# Run without transcript control for comparison
!python src/experiment.py --stage all --model microsoft/phi-2 --n-items 50
```

Compare probe AUCs:
- If similar → stronger evidence for true trust encoding
- If much lower → likely transcript features

### Step 3: Pythia-1B with Transcript Control

```python
!python src/experiment.py --stage all --model EleutherAI/pythia-1b --n-items 50 --neutral-responses
```

**Key Test**: Pythia had AUC 0.80 but 0% behavior. If AUC drops with neutral responses → confirms transcript feature hypothesis. If AUC stays high → suggests true encoding that's just not used behaviorally.

---

## What to Look For

### Good Signs (Supports Trust Hypothesis)

1. **Probe AUC stays high** with neutral responses (0.6+)
2. **Behavioral effects remain** with statistical significance (p < 0.05)
3. **Probe scores change** in expected direction under steering
4. **Epistemic pattern**: High-trust accepts TRUE corrections more than FALSE

### Warning Signs (Suggests Transcript Features)

1. **Probe AUC drops** significantly with neutral responses
2. **Behavioral effects disappear** with neutral responses
3. **Probe scores don't change** as expected under steering
4. **No epistemic pattern**: High-trust accepts everything more (deference)

---

## Expected Runtime

- **Phi-2**: ~30-45 minutes (2.7B parameters)
- **GPT-2**: ~10-15 minutes (124M parameters)
- **Pythia-1B**: ~20-30 minutes (1B parameters)

---

## Output Files

All results saved to `data/results/`:
- `behavioral_results.csv`: Individual conversation metrics
- `behavioral_stats.json`: Summary with CIs and p-values
- `probe_results.json`: Probe training results
- `intervention_results.json`: Steering results with probe scores
- `behavioral_plots.png`: Visualization

---

## Quick Check Commands

```python
# View behavioral stats with CIs
import json
with open('data/results/behavioral_stats.json') as f:
    stats = json.load(f)
    print(f"Difference: {stats['ur_difference']:.3f}")
    print(f"95% CI: [{stats['ur_difference_ci_lower']:.3f}, {stats['ur_difference_ci_upper']:.3f}]")
    print(f"p-value: {stats['ur_difference_p_value']:.4f}")

# View probe results
with open('data/results/probe_results.json') as f:
    probes = json.load(f)
    for layer, metrics in probes['probe_metrics'].items():
        if 'test_auc' in metrics:
            print(f"{layer}: AUC = {metrics['test_auc']:.3f}")

# View intervention results
with open('data/results/intervention_results.json') as f:
    intervention = json.load(f)
    for strength, data in sorted(intervention.items()):
        print(f"Strength {strength}: UR={data['mean_update_rate']:.3f}, Probe={data.get('mean_probe_score', 'N/A')}")
```

---

## Troubleshooting

**If you get import errors**:
```python
!pip install scipy
```

**If model loading fails**:
- Check HuggingFace authentication for gated models
- Try smaller model first (GPT-2) to test setup

**If memory errors**:
- Reduce `--n-items` to 25
- Run stages separately instead of `--stage all`

