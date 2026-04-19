# ============================================================
#  BUILD INSTRUCTIONS — Topological Photonic Lattice Explorer
# ============================================================

## Prerequisites
Install the required packages (once):

    pip install pyinstaller PyQt5 matplotlib numpy scipy

If you also want JAX acceleration (optional):

    pip install jax jaxlib          # CPU
    # or for NVIDIA GPU:
    pip install jax[cuda12]         # adjust to your CUDA version

---

## Files needed in the SAME folder

    Linear.py                  ← main application (entry point)
    NonLinear.py               ← component, loaded at runtime by Linear.py
    original.png               ← background / splash image
    icon.ico                   ← window icon
    TopologicalPhotonic.spec   ← PyInstaller build recipe  ← this file

---

## Build the .exe  (run once, takes 1-3 minutes)

    pyinstaller TopologicalPhotonic.spec

Output:  dist\TopologicalPhotonicExplorer.exe

The .exe is fully self-contained — no Python installation needed on the target machine.

---

## Quick one-liner alternative (no .spec file needed)

    pyinstaller --onefile --windowed --icon=icon.ico ^
        --add-data "NonLinear.py;." ^
        --add-data "original.png;." ^
        --add-data "icon.ico;." ^
        --name TopologicalPhotonicExplorer ^
        Linear.py

(On Mac/Linux replace the ^ line-continuation with \ and use : instead of ; in --add-data)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No module named jax" at runtime | JAX not installed — that's fine, the app falls back to NumPy automatically |
| Blank window / Qt platform error | Run `pip install pyqt5` again; make sure Qt5Agg is the matplotlib backend |
| App crashes silently | Remove `console=False` from the .spec (or add `--console`) to see error output |
| Icon not showing on Windows | Make sure icon.ico contains a 256×256 layer (the one we generated does) |
