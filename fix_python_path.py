"""
Fix Python path for PyInstaller executables
This should be imported first in any entry point
"""
import sys
from pathlib import Path

def fix_python_path():
    """Fix sys.path for PyInstaller executables"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        if hasattr(sys, '_MEIPASS'):
            # Onefile mode - use MEIPASS
            sys.path.insert(0, sys._MEIPASS)
        elif sys.platform == 'darwin':
            # macOS app bundle - onedir mode
            # Executable is in Contents/MacOS/
            # Modules are in Contents/Resources/
            if '.app/Contents/MacOS' in sys.executable:
                app_bundle = Path(sys.executable).parent.parent
                resources = app_bundle / 'Resources'
                if resources.exists():
                    # Add Resources to path (where all modules are)
                    if str(resources) not in sys.path:
                        sys.path.insert(0, str(resources))
                    # Also check for site-packages in Resources
                    site_packages = resources / 'site-packages'
                    if site_packages.exists() and str(site_packages) not in sys.path:
                        sys.path.insert(0, str(site_packages))
        else:
            # Other platforms
            exe_dir = Path(sys.executable).parent
            if str(exe_dir) not in sys.path:
                sys.path.insert(0, str(exe_dir))

# Auto-fix on import
fix_python_path()

