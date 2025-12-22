# Critical Concerns and Proposed Fixes

## Summary of Reviewer Feedback

A knowledgeable reviewer (familiar with Neel Nanda's work) provided critical feedback identifying potential confounds and suggesting improvements. This document addresses those concerns and outlines implementation plans.

---

## Valid Concerns Identified

### 1. **Label Leakage / "Transcript Features" Confound**

**Concern**: Probes might be detecting superficial transcript features rather than a true "latent trust state."

**Potential Leakage Channels**:
- ✅ **Assistant wording**: Actually CONTROLLED - assistant always says "I see. Thank you for the correction." in both conditions
- ⚠️ **Number token distributions**: Correct vs incorrect answers have different digit patterns, punctuation, etc.
- ⚠️ **Full history visible**: Extraction point is after seeing entire conversation, so probe can "remember" any superficial marker

**Evidence Supporting Concern**:
- Pythia shows huge AUC (0.80) but 0% behavioral effect - exactly what you'd expect if probe reads transcript features not used by behavior mechanism

### 2. **Statistical Rigor**

**Concern**: Small effect sizes (5-6%) with n=50 per condition need proper statistical tests.

**Missing**:
- Confidence intervals for ΔUR
- Paired significance tests (McNemar's or bootstrap)
- Effect size estimates

### 3. **Stratification by Final Correction Truth**

**Concern**: Need to check if "trust" is epistemically sensible (accepts true things more) or just deference (accepts everything more).

**Current Status**: We DO compute this but should emphasize it more in analysis.

### 4. **Phi-2 Steering Direction Confusion**

**Concern**: Negative steering increases acceptance - this seems backwards. Could be:
- Sign convention issue
- Probe direction not aligned with behavior
- Steering hitting different feature (politeness/deference vs trust)

---

## What We Can Already Claim (Safely)

### 1. Highly Decodable Condition Signal
- Probe AUC 0.7-0.8 means hidden states contain enough information to classify high-trust vs low-trust script
- This is a useful empirical fact, even if it's "just transcript features"

### 2. Behavioral Effects (with caveats)
- GPT-2 (+6%) and Phi-2 (+5%) show history dependence
- Need statistical tests to confirm these aren't noise

### 3. Phi-2 Internal→Behavioral Link
- Probe-behavior correlation: 0.12-0.17 (nontrivial)
- Steering produces large behavior change with dose-response
- This is the most "Neel-stream relevant" finding

---

## Proposed Fixes (High Value, Low Effort)

### Fix A: Add Statistical Tests

**Implementation**:
1. Paired bootstrap over base items (since we have matched pairs)
2. 95% confidence intervals for ΔUR
3. McNemar's test on accept/reject (treat ambiguous separately)

**Code to Add**:
```python
# In metrics.py
import scipy.stats
from scipy.stats import mcnemar
import numpy as np

def compute_statistical_tests(metrics_list):
    """Add proper statistical tests."""
    df = metrics_to_dataframe(metrics_list)
    
    # Paired bootstrap for ΔUR
    high_trust_ur = df[df["condition"] == "high_trust"]["update_rate"].values
    low_trust_ur = df[df["condition"] == "low_trust"]["update_rate"].values
    
    # Bootstrap confidence interval
    n_bootstrap = 10000
    diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(high_trust_ur), len(high_trust_ur), replace=True)
        diff = high_trust_ur[indices].mean() - low_trust_ur[indices].mean()
        diffs.append(diff)
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    # McNemar's test (convert to binary accept/reject)
    # ...
```

### Fix B: Emphasize Stratification by Final Correction Truth

**Current Status**: We already compute this! But need to:
1. Make it more prominent in output
2. Interpret what it means
3. Check if trust is "epistemically sensible"

**What to Report**:
- UR_high, final_true vs UR_high, final_false
- UR_low, final_true vs UR_low, final_false
- Does high-trust accept TRUE corrections more? (epistemically sensible)
- Or does it accept EVERYTHING more? (just deference)

### Fix C: Transcript Control - Equalize Assistant Wording

**Current Issue**: Assistant says same thing in both conditions ("I see. Thank you for the correction."), BUT:
- The numbers themselves differ (correct vs wrong answers)
- This creates token distribution differences

**Proposed Control**:
1. Use identical neutral template for ALL assistant responses
2. Or: Use same numbers but swap which are "correct" based on condition
3. Or: Extract hidden states BEFORE the model sees assistant's history responses

**Implementation**:
```python
# Option 1: More neutral template
turns.append({
    "role": "assistant",
    "content": "Noted. Let's continue."  # No "thank you" signal
})

# Option 2: Extract at different position
# Extract at position BEFORE assistant responses in history
# Only see user corrections, not assistant acknowledgments
```

---

## Revised Narrative (Per Reviewer Suggestion)

### Instead of: "All models have a trust variable"

### Frame as: "Representation-behavior dissociation + causal handle in Phi-2"

**Story**:
1. Many models contain a **decodable state** about user's past correctness
2. In GPT-NeoX (Pythia), this state appears **not to be used** for final accept/reject decision
3. In Phi-2, state is **more behaviorally entangled**, and we have a **causal knob** (steering)

**Why This Framing is Better**:
- Model-biology flavored (compare architectures)
- Pragmatic (causal control / monitoring relevance)
- Acknowledges the dissociation rather than claiming universal trust variable

---

## Implementation Priority

### Immediate (Before Submission)

1. ✅ **Statistical Tests** - Add CIs and significance tests
2. ✅ **Stratification Analysis** - Emphasize final_correction_true breakdown
3. ✅ **Steering Direction Check** - Verify probe scores change in expected direction under steering

### High Value Next Steps

4. **Transcript Control** - Equalize assistant wording or extract at different position
5. **Probe Score Under Steering** - Check if steering changes probe scores as expected
6. **Killer Figure** - 4-panel plot showing:
   - ΔUR with 95% CI (per model)
   - Best probe AUC (per model)
   - Probe score → acceptance curve (Phi-2 vs Pythia)
   - Phi-2 steering dose-response + probe score shift

---

## Current Assistant Wording (Already Controlled!)

**Good News**: We already use identical assistant wording in both conditions:

```python
# Both high-trust and low-trust get:
turns.append({
    "role": "assistant",
    "content": f"I see. Thank you for the correction."
})
```

**However**, leakage could still come from:
- Different number tokens (correct vs wrong answers)
- Different conversation lengths (if corrections differ)
- Token distribution patterns in user corrections

---

## Next Steps

1. **Add statistical tests** to metrics.py
2. **Create transcript control version** of dataset generator
3. **Add probe score tracking** during steering experiments
4. **Create comprehensive visualization** (4-panel figure)
5. **Update project summary** with these concerns and fixes

