# HuggingFace Authentication for Gated Models

## Step 1: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy the token

## Step 2: Authenticate in Colab

Run this in Colab:

```python
from huggingface_hub import login

# This will prompt you to paste your token
login()

# Or set it directly (less secure but convenient)
# login(token="your_token_here")
```

## Step 3: Accept Model Terms

1. Visit https://huggingface.co/google/gemma-2-2b
2. Click "Agree and access repository"
3. Then you can use the model

## Step 4: Run Experiments

```python
!python src/experiment.py --stage all --model google/gemma-2-2b --n-items 50
```

