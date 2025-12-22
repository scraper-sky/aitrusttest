# Quick Start for Google Colab

## Step 1: Clone and Setup

Run this in a Colab cell:

```python
# Clone the repository
!git clone https://github.com/scraper-sky/aitrusttest.git

# Navigate to the project
import os
os.chdir('/content/aitrusttest')

# Verify you're in the right place
!pwd
!ls -la
```

## Step 2: Install Dependencies

```python
!pip install -r requirements.txt
```

## Step 3: Run Experiments

```python
# Generate datasets
!python src/experiment.py --stage generate --model gpt2 --n-items 50

# Run behavioral experiments
!python src/experiment.py --stage behavioral --model gpt2

# Train probes (optional)
!python src/experiment.py --stage probe --model gpt2

# Run interventions (optional)
!python src/experiment.py --stage intervention --model gpt2
```

## Troubleshooting

If you get "No such file or directory" errors:

1. **Make sure you're in the right directory:**
   ```python
   import os
   print(os.getcwd())  # Should show /content/aitrusttest
   ```

2. **If directory doesn't exist, clone again:**
   ```python
   !rm -rf aitrusttest
   !git clone https://github.com/scraper-sky/aitrusttest.git
   %cd aitrusttest
   ```

3. **Pull latest changes if you already cloned:**
   ```python
   %cd /content/aitrusttest
   !git pull
   ```

## Enable GPU

Before running, make sure GPU is enabled:
- Runtime → Change runtime type → GPU → Save

## Check GPU

```python
!nvidia-smi
```

