# Running Earned Trust Experiments on Google Colab

## Quick Setup

1. **Upload your project to Colab:**
   - Option A: Upload the entire `aitrusttest` folder to Google Drive
   - Option B: Clone from GitHub (if you push it)
   - Option C: Upload files directly to Colab

2. **In Colab, run:**

```python
# Mount Google Drive (if files are there)
from google.colab import drive
drive.mount('/content/drive')

# Or clone from GitHub
!git clone https://github.com/scraper-sky/aitrusttest.git

# Navigate to project
import os
# If using GitHub (recommended):
os.chdir('/content/aitrusttest')
# OR if using Google Drive:
# os.chdir('/content/drive/MyDrive/aitrusttest')

# Install dependencies
!pip install torch transformers numpy pandas scikit-learn matplotlib seaborn tqdm accelerate datasets

# Run experiments
!python src/experiment.py --stage generate --model gpt2 --n-items 50
!python src/experiment.py --stage behavioral --model gpt2
```

## Using GPU

Make sure to enable GPU in Colab:
- Runtime → Change runtime type → GPU (T4 or better with Pro)

The code will automatically detect and use GPU if available.

## Recommended Models for Colab Pro

- **Small/Testing**: `gpt2`, `gpt2-medium`
- **Medium**: `gpt2-large`, `microsoft/DialoGPT-medium`
- **Larger (if you have enough memory)**: `meta-llama/Llama-2-7b-hf` (requires HuggingFace auth)

## Tips

1. **Save results to Drive** so they persist after session ends
2. **Use smaller batches** if you run out of memory
3. **Monitor GPU usage** with `!nvidia-smi`

