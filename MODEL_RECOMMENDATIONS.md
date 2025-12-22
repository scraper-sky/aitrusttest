# Model Recommendations for Mechanistic Interpretability Research

## Best Models for MAT/Mechanistic Interpretability

### Tier 1: Most Recommended (Best for Research)

#### 1. **Pythia Models (EleutherAI)** ⭐ BEST CHOICE
- **Why**: Specifically designed for interpretability research
- **Models**: 
  - `EleutherAI/pythia-160m` (small, fast)
  - `EleutherAI/pythia-410m` (medium)
  - `EleutherAI/pythia-1b` (larger, good balance)
  - `EleutherAI/pythia-2.8b` (if you have memory)
- **Pros**: 
  - Open source, no auth needed
  - Training data documented
  - Checkpoints at different training steps
  - Widely used in mech interp research
- **Command**: 
  ```python
  !python src/experiment.py --stage all --model EleutherAI/pythia-1b --n-items 50
  ```

#### 2. **GPT-2 Medium/Large**
- **Why**: Most studied model in mech interp, well-understood
- **Models**:
  - `gpt2-medium` (355M) - good balance
  - `gpt2-large` (774M) - if you have memory
- **Pros**:
  - Extensive research literature
  - Well-documented behavior
  - Easy to compare with existing work
- **Command**:
  ```python
  !python src/experiment.py --stage all --model gpt2-medium --n-items 50
  ```

### Tier 2: Good Alternatives

#### 3. **TinyStories Models**
- **Why**: Small, interpretable, designed for analysis
- **Models**: `roneneldan/TinyStories-1M` or `roneneldan/TinyStories-33M`
- **Pros**: Very small, fast, easy to analyze
- **Cons**: May be too simple for complex trust effects

#### 4. **Phi-2 (Microsoft)**
- **Why**: Open, good size, modern architecture
- **Model**: `microsoft/phi-2`
- **Pros**: 2.7B params, open, good performance
- **Cons**: Less studied in mech interp literature

### Tier 3: If You Need Larger Models

#### 5. **LLaMA-2 7B** (if you can authenticate)
- **Why**: Larger, more capable
- **Model**: `meta-llama/Llama-2-7b-hf`
- **Cons**: Requires authentication, very large

## Recommended for Your Trust Experiment

### Primary Recommendation: **Pythia-1B**
```python
!python src/experiment.py --stage all --model EleutherAI/pythia-1b --n-items 50
```

**Why Pythia-1B?**
1. ✅ Perfect size for analysis (1B params - not too small, not too large)
2. ✅ Designed for interpretability research
3. ✅ Open source, no authentication
4. ✅ Widely used in mech interp community
5. ✅ Good balance of capability and analyzability
6. ✅ Checkpoints available for training dynamics analysis

### Secondary: **GPT-2 Medium**
```python
!python src/experiment.py --stage all --model gpt2-medium --n-items 50
```

**Why GPT-2 Medium?**
1. ✅ Most studied model in mech interp
2. ✅ Easy to compare with existing research
3. ✅ Well-documented behavior
4. ✅ Good size for analysis

## Size vs. Capability Trade-off

- **Small (160M-410M)**: Fast, easy to analyze, but may show weaker effects
- **Medium (1B-2.8B)**: ⭐ **Sweet spot** - good effects, still analyzable
- **Large (7B+)**: Strong effects, but harder to analyze, more memory

## For MAT Specifically

If you're submitting to MAT or Neel's framework:
- **Pythia-1B** or **Pythia-2.8B** are excellent choices
- **GPT-2 Medium/Large** are also great (most common in mech interp)
- Both are well-supported in common mech interp tools

