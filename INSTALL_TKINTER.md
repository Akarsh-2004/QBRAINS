# Installing Tkinter on macOS

If you want to use the tkinter-based desktop app (`desktop_app.py`), you need to install tkinter support.

## Option 1: Use System Python (Recommended for tkinter)

System Python on macOS includes tkinter by default:

```bash
# Use system Python instead of Homebrew Python
/usr/bin/python3 desktop_app.py
```

## Option 2: Install Python with tkinter via Homebrew

```bash
# Install Python with tkinter support
brew install python-tk

# Or reinstall Python with tkinter
brew reinstall python@3.13
```

## Option 3: Use pyenv with tkinter

```bash
# Install pyenv
brew install pyenv

# Install Python with tkinter
env PYTHON_CONFIGURE_OPTS="--with-tcltk" pyenv install 3.13.5
```

## Option 4: Use the Web-based Desktop App (No tkinter needed!)

The web-based desktop app (`desktop_app_web.py`) doesn't require tkinter:

```bash
python desktop_app_web.py
```

This opens the web dashboard in your browser - same functionality, no tkinter dependency!

## Recommended Solution

**Use `desktop_app_web.py`** - it's easier and works on all platforms without additional setup.

