# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `aitrusttest` (or your preferred name)
3. Choose **Public** or **Private**
4. **DO NOT** check "Initialize with README" (we already have files)
5. Click **"Create repository"**

## Step 2: Connect and Push

After creating the repo, GitHub will show you commands. Use these (replace `YOUR_USERNAME` with your GitHub username):

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/aitrusttest.git

# Rename branch to 'main' (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Clone on Colab

Once pushed, you can clone it on Colab:

```python
!git clone https://github.com/YOUR_USERNAME/aitrusttest.git
%cd aitrusttest
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/aitrusttest.git
git branch -M main
git push -u origin main
```

