# Virtual Environment Setup Guide

A virtual environment has been created for your project to avoid conflicts with system Python packages.

## ✅ Installation Complete

All required packages have been installed:
- ✅ PyTorch (torch, torchvision, torchaudio)
- ✅ Transformers (HuggingFace)
- ✅ Datasets
- ✅ Scikit-learn

## 🔧 Using the Virtual Environment

### Option 1: Terminal/Command Line

```bash
# Activate virtual environment
cd /Users/ayushjd-fusion/Desktop/QBRAINS
source venv/bin/activate

# Now run your scripts
python train_emotion_llm.py

# Or run Python
python
```

### Option 2: Jupyter Notebook

To use the virtual environment in Jupyter:

```bash
# Activate virtual environment
source venv/bin/activate

# Install ipykernel
pip install ipykernel

# Add virtual environment to Jupyter
python -m ipykernel install --user --name=qbrains --display-name "Python (QBRAINS)"

# Start Jupyter
jupyter notebook
```

Then in Jupyter:
1. Open your notebook
2. Go to **Kernel** → **Change Kernel** → **Python (QBRAINS)**

### Option 3: VS Code / Cursor

1. Open the project folder
2. VS Code/Cursor should detect the virtual environment automatically
3. Select the Python interpreter: `venv/bin/python`
4. Or press `Cmd+Shift+P` → "Python: Select Interpreter" → Choose `venv/bin/python`

## 📝 Quick Test

Test if everything works:

```bash
source venv/bin/activate
python -c "import torch; import transformers; print('✅ All packages installed!')"
```

## 🚀 Next Steps

1. **Activate the virtual environment** (see above)
2. **Run the training script**: `python train_emotion_llm.py`
3. **Or use the notebook**: `jupyter notebook notebooks/llm_emotion_training.ipynb`

## 💡 Tips

- Always activate the virtual environment before running scripts
- The virtual environment is in `venv/` folder
- Add `venv/` to `.gitignore` if using git
- To deactivate: `deactivate`

## 🔄 If You Need to Reinstall

```bash
# Remove old virtual environment
rm -rf venv

# Create new one
python3 -m venv venv

# Activate and install
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio transformers datasets
```

