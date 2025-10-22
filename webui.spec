# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import gradio
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ---- Collect all package data and submodules ----
datas = []
hiddenimports = []
binaries = [
    ('./ffmpeg.exe', '.'),
    ('./ffprobe.exe', '.'),
	('./model_orig.pkl', '.')
]

# Include all Gradio-related packages
for pkg in ["gradio", "gradio_client", "safehttpx", "groovy", "httpx", "requests", "sklearn", "tabulate"]:
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

# ---- Force-bundle *every* .py file under gradio/ ----
gradio_root = Path(gradio.__file__).parent
for f in gradio_root.rglob("*.py"):
    rel = f.relative_to(gradio_root)
    dest = f"gradio/{rel.parent}" if rel.parent != Path() else "gradio"
    datas.append((str(f), dest))
	

a = Analysis(
    ['webui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ett-vid-clipper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
