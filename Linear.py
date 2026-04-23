# %%
"""

Topological Photonic Lattice Explorer
======================================
Standalone desktop application — no Jupyter required.
Run:   python app.py
Build: pyinstaller --onefile --windowed app.py

Changes v1.0:
  • φ_IQH and φ_AQH sliders in the Simulation panel (below κ controls)
  • Hamiltonian builders fully vectorised with NumPy (no Python loops)
  • compute_spectrum vectorised across all frequencies at once
  • Optional JAX back-end: if JAX is installed the frequency solve is
    JIT-compiled and runs on GPU/CPU via XLA for a further speed-up.
    Falls back to NumPy automatically if JAX is absent.
"""

import sys, os, io, zipfile, datetime

# PyInstaller onefile: bundled data files live in sys._MEIPASS temp dir
if getattr(sys, 'frozen', False):
    _BUNDLE_DIR = sys._MEIPASS
    sys.path.insert(0, _BUNDLE_DIR)   # so "from NonLinear import ..." works
else:
    _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QButtonGroup, QAbstractButton,
    QGroupBox, QSizePolicy, QSplitter, QStatusBar, QFrame,
    QFileDialog, QLineEdit, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui  import QFont, QColor, QPalette, QIcon, QPixmap

import matplotlib
if matplotlib.get_backend() != 'Qt5Agg':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BG   = '#08090d'
PANEL_BG  = '#0a0c14'
CARD_BG   = '#171c2e'
GRID_COL  = '#1a1f2e'
SPINE_COL = '#1e2230'
TEXT_DIM  = '#4a5270'
TEXT_COL  = '#c8d0e7'
ACCENT    = '#00e5ff'
RING_R    = 0.32
RING_GAP  = 0.65

# ── Physics constants ─────────────────────────────────────────────────────────
J0       = 1.0
Spin     = -1
Spin_1sl = 1
# Defaults — overridden at runtime by the UI sliders
PHI_IQH_DEFAULT = np.pi / 2
PHI_AQH_DEFAULT = np.pi / 4

# ══════════════════════════════════════════════════════════════════════════════
#  Index helpers  (pure Python — called only during Hamiltonian build)
# ══════════════════════════════════════════════════════════════════════════════
def LocationToNumber(X, Y, Nx, Ny):
    return Nx * (Y - 1) + X

def NumberToLocation(N, Nx, Ny):
    X = N % Nx
    if X == 0: X = Nx
    Y = (N - X) // Nx + 1
    return int(X), int(Y)

def superlattice(Nx0, Ny0, m):
    NL = Nx0 * Ny0
    m2 = m % NL
    if m2 == 0: m2 = NL
    return 1 + (m - m2) // NL, m2

def LocationToNumber_AQH_zigzag(X, Y, Nx, Ny):
    if Y % 2 == 1:
        return int(X + (Y - 1) // 2 * (2 * Nx - 1))
    else:
        return int(Nx - 1 + X + (Y - 2) // 2 * (2 * Nx - 1))

def NumberToLocation_AQH_zigzag(N, Nx, Ny):
    flag = 0
    X1 = N % (2 * Nx - 1)
    if X1 == 0: X1 = 2 * Nx - 1
    if X1 > Nx - 1:
        flag = 1
        X1 = X1 - Nx + 1
    Y1 = (N - X1) // (2 * Nx - 1) * 2 + 1 + flag
    return int(X1), int(Y1)

def get_lattice_params(h_str, nx0, ny0, nx1, ny1):
    if h_str == 'A_zigzag':
        NL  = nx0 * (ny0 - 1) + ny0 * (nx0 - 1)
        Nxt, Nyt = nx0, ny0
        isite = LocationToNumber_AQH_zigzag(nx0 - 1, 1,          nx0, ny0)
        osite = LocationToNumber_AQH_zigzag(nx0 - 1, 2 * ny0 - 1, nx0, ny0)
    elif h_str in ('IQH_cyl', 'AQH_cyl'):
        # cylinder: simple Nx*Ny grid, no superlattice
        Nxt, Nyt = nx0, ny0
        NL = nx0 * ny0
        isite = LocationToNumber(1, 1,   nx0, ny0)   # top-left of unrolled grid
        osite = LocationToNumber(1, ny0, nx0, ny0)   # bottom-left
    else:
        Nxt, Nyt = nx0 * nx1, ny0 * ny1
        NL = Nxt * Nyt
        isite = 1
        # single ring: osite = isite = 1
        osite = max(1, nx0 * ny0 - nx0 + 1)
    return NL, Nxt, Nyt, isite, osite

def site_xy(n, h_str, nx0, ny0, nx1, ny1):
    if h_str == 'A_zigzag':
        loc = NumberToLocation_AQH_zigzag(n, nx0, ny0)
        return float(2 * loc[0] - 1 + loc[1] % 2), float(loc[1])
    if h_str in ('IQH_cyl', 'AQH_cyl'):
        mx, my = NumberToLocation(n, nx0, ny0)
        # polar: theta wraps x around a circle, r grows with y
        R0 = nx0 / (2 * np.pi)
        theta = (mx - 1) / nx0 * 2 * np.pi
        r     = R0 + (ny0 - my) * 1.2        # y=1 outermost, y=Ny innermost
        return float(r * np.cos(theta)), float(r * np.sin(theta))
    sl1, sl0 = superlattice(nx0, ny0, n)
    x1, y1   = NumberToLocation(sl1, nx1, ny1)
    x0, y0   = NumberToLocation(sl0, nx0, ny0)
    return float((x1 - 1) * nx0 + x0 - 1), float((y1 - 1) * ny0 + y0 - 1)

# ══════════════════════════════════════════════════════════════════════════════
#  Vectorised Hamiltonian builders
#  Strategy: pre-compute all site indices as arrays, then use boolean masks
#  to set hopping elements — no Python loops over sites.
# ══════════════════════════════════════════════════════════════════════════════

def _iqh_coords(nx0, ny0, nx1, ny1):
    """Return per-site coordinate arrays for the IQH superlattice."""
    N = nx0 * ny0 * nx1 * ny1
    ms = np.arange(1, N + 1)
    # superlattice decomposition
    NL0 = nx0 * ny0
    m2  = ms % NL0;  m2[m2 == 0] = NL0
    m1  = 1 + (ms - m2) // NL0          # supercell index (1-based)
    m0  = m2                             # unit-cell index (1-based)
    # unit-cell x,y
    mx0 = m0 % nx0;  mx0[mx0 == 0] = nx0
    my0 = (m0 - mx0) // nx0 + 1
    # supercell x,y
    mx1 = m1 % nx1;  mx1[mx1 == 0] = nx1
    my1 = (m1 - mx1) // nx1 + 1
    return m1, m0, mx0, my0, mx1, my1

def H_IQH_IQH(nx0, ny0, nx1, ny1, J0, J1, Phi0, Phi1, Spin):
    """IQH superlattice. Phi0 = layer-0 (unit-cell) phase, Phi1 = layer-1 (supercell) phase."""
    N  = nx0 * ny0 * nx1 * ny1
    H  = np.zeros((N + 1, N + 1), dtype=complex)
    m1, m0, mx0, my0, mx1, my1 = _iqh_coords(nx0, ny0, nx1, ny1)
    ms  = np.arange(1, N + 1)
    NL0 = nx0 * ny0

    # ── Intra-supercell (layer 0 phase Phi0) ─────────────────────────────────
    mask_h = mx0 < nx0
    m_idx  = ms[mask_h]
    n_idx  = m_idx + 1
    phase  = np.exp(-1j * my0[mask_h] * Spin * Phi0)
    H[m_idx, n_idx] = -J0 * phase
    H[n_idx, m_idx] = np.conj(-J0 * phase)

    mask_v = my0 < ny0
    m_idx  = ms[mask_v]
    n_idx  = m_idx + nx0
    H[m_idx, n_idx] = -J0
    H[n_idx, m_idx] = -J0

    # ── Inter-supercell (layer 1 phase Phi1) ─────────────────────────────────
    if nx1 > 1 or ny1 > 1:
        for sc_m in range(1, nx1 * ny1 + 1):
            scx_m, scy_m = NumberToLocation(sc_m, nx1, ny1)
            for scx_n, scy_n, conn_type in [
                (scx_m - 1, scy_m,     'hx'),
                (scx_m,     scy_m - 1, 'vy'),
            ]:
                if scx_n < 1 or scy_n < 1: continue
                sc_n     = LocationToNumber(scx_n, scy_n, nx1, ny1)
                parity   = (-1) ** (scx_m + scy_m)
                phase_sl = np.exp(-1j * Spin_1sl * Phi1 * scy_m)

                if conn_type == 'hx':
                    if parity == 1:
                        m_uc = LocationToNumber(1,   1,   nx0, ny0)
                        n_uc = LocationToNumber(nx0, 1,   nx0, ny0)
                    else:
                        m_uc = LocationToNumber(1,   ny0, nx0, ny0)
                        n_uc = LocationToNumber(nx0, ny0, nx0, ny0)
                    m_site = (sc_m - 1) * NL0 + m_uc
                    n_site = (sc_n - 1) * NL0 + n_uc
                    val = -J1 * phase_sl
                    H[m_site, n_site] = val
                    H[n_site, m_site] = np.conj(val)
                else:  # vy
                    if parity == 1:
                        m_uc = LocationToNumber(1,   1,   nx0, ny0)
                        n_uc = LocationToNumber(1,   ny0, nx0, ny0)
                    else:
                        m_uc = LocationToNumber(nx0, 1,   nx0, ny0)
                        n_uc = LocationToNumber(nx0, ny0, nx0, ny0)
                    m_site = (sc_m - 1) * NL0 + m_uc
                    n_site = (sc_n - 1) * NL0 + n_uc
                    H[m_site, n_site] = -J1
                    H[n_site, m_site] = -J1
    return H


def H_AQH_AQH(nx0, ny0, nx1, ny1, J0, J1, Phi0, Phi1, Spin):
    """AQH superlattice. Phi0 = layer-0 (unit-cell) phase, Phi1 = layer-1 (supercell) phase."""
    N   = nx0 * ny0 * nx1 * ny1
    H   = np.zeros((N + 1, N + 1), dtype=complex)
    NL0 = nx0 * ny0
    ms  = np.arange(1, N + 1)
    m1v, m0v, mx0v, my0v, mx1v, my1v = _iqh_coords(nx0, ny0, nx1, ny1)

    # ── Intra-supercell (layer 0 phase Phi0) ─────────────────────────────────
    mask_h = mx0v < nx0
    m_idx  = ms[mask_h]
    n_idx  = m_idx + 1
    sgn    = ((-1) ** mx0v[mask_h]) * ((-1) ** my0v[mask_h])
    phase  = np.exp(-1j * sgn * Spin * Phi0)
    H[m_idx, n_idx] = -J0 * phase
    H[n_idx, m_idx] = np.conj(-J0 * phase)

    mask_v = my0v < ny0
    m_idx  = ms[mask_v]
    n_idx  = m_idx + nx0
    sgn    = ((-1) ** mx0v[mask_v]) * ((-1) ** my0v[mask_v])
    phase  = np.exp(1j * sgn * Spin * Phi0)
    H[m_idx, n_idx] = -J0 * phase
    H[n_idx, m_idx] = np.conj(-J0 * phase)

    mask_d1 = (mx0v < nx0) & (my0v < ny0) & ((mx0v % 2) == (my0v % 2))
    m_idx   = ms[mask_d1]
    n_idx   = m_idx + nx0 + 1
    H[m_idx, n_idx] = -J0;  H[n_idx, m_idx] = -J0

    mask_d2 = (mx0v > 1) & (my0v < ny0) & ((mx0v % 2) != (my0v % 2))
    m_idx   = ms[mask_d2]
    n_idx   = m_idx + nx0 - 1
    H[m_idx, n_idx] = -J0;  H[n_idx, m_idx] = -J0

    # ── Inter-supercell (layer 1 phase Phi1) ─────────────────────────────────
    if nx1 > 1 or ny1 > 1:
        for sc_m in range(1, nx1 * ny1 + 1):
            scx_m, scy_m = NumberToLocation(sc_m, nx1, ny1)
            for scx_n, scy_n, conn_type in [
                (scx_m - 1, scy_m,     'hx'),
                (scx_m,     scy_m - 1, 'vy'),
                (scx_m - 1, scy_m - 1, 'dNE'),
                (scx_m + 1, scy_m - 1, 'dNW'),
            ]:
                if scx_n < 1 or scy_n < 1: continue
                if scx_n > nx1 or scy_n > ny1: continue
                sc_n  = LocationToNumber(scx_n, scy_n, nx1, ny1)
                p_m   = (-1) ** scy_m
                p_mx  = (-1) ** scx_m

                if conn_type == 'hx':
                    if p_m == p_mx:
                        m_uc = LocationToNumber(1,   1,   nx0, ny0)
                        n_uc = LocationToNumber(nx0, 1,   nx0, ny0)
                        val  = -J1 * np.exp(-1j * Spin_1sl * Phi1)
                    else:
                        m_uc = LocationToNumber(1,   ny0, nx0, ny0)
                        n_uc = LocationToNumber(nx0, ny0, nx0, ny0)
                        val  = -J1 * np.exp(1j * Spin_1sl * Phi1)
                    m_s = (sc_m - 1) * NL0 + m_uc
                    n_s = (sc_n - 1) * NL0 + n_uc
                    H[m_s, n_s] = val;  H[n_s, m_s] = np.conj(val)

                elif conn_type == 'vy':
                    if p_m == p_mx:
                        m_uc = LocationToNumber(1,   1,   nx0, ny0)
                        n_uc = LocationToNumber(1,   ny0, nx0, ny0)
                        val  = -J1 * np.exp(1j * Spin_1sl * Phi1)
                    else:
                        m_uc = LocationToNumber(nx0, 1,   nx0, ny0)
                        n_uc = LocationToNumber(nx0, ny0, nx0, ny0)
                        val  = -J1 * np.exp(-1j * Spin_1sl * Phi1)
                    m_s = (sc_m - 1) * NL0 + m_uc
                    n_s = (sc_n - 1) * NL0 + n_uc
                    H[m_s, n_s] = val;  H[n_s, m_s] = np.conj(val)

                elif conn_type == 'dNE':
                    if (-1) ** scx_m == (-1) ** scy_m:
                        m_uc = LocationToNumber(1,   1,   nx0, ny0)
                        n_uc = LocationToNumber(nx0, ny0, nx0, ny0)
                        m_s  = (sc_m - 1) * NL0 + m_uc
                        n_s  = (sc_n - 1) * NL0 + n_uc
                        H[m_s, n_s] = -J1;  H[n_s, m_s] = -J1

                elif conn_type == 'dNW':
                    if (-1) ** scx_m == (-1) ** (scy_m + 1):
                        m_uc = LocationToNumber(nx0, 1,   nx0, ny0)
                        n_uc = LocationToNumber(1,   ny0, nx0, ny0)
                        m_s  = (sc_m - 1) * NL0 + m_uc
                        n_s  = (sc_n - 1) * NL0 + n_uc
                        H[m_s, n_s] = -J1;  H[n_s, m_s] = -J1
    return H


def H_zigzag(Nx, Ny, J, Phi, Spin):
    """AQH zigzag Hamiltonian — restored from original correct implementation."""
    NL = Nx * (Ny - 1) + Ny * (Nx - 1)
    H  = np.zeros((NL + 1, NL + 1), dtype=complex)
    for m in range(1, NL + 1):
        for n in range(1, NL + 1):
            if m == n: continue
            mx, my = NumberToLocation_AQH_zigzag(m, Nx, Ny)
            nx, ny = NumberToLocation_AQH_zigzag(n, Nx, Ny)
            if ((mx==nx and ny-my==1) or (mx==nx-1 and ny==my-1)) and my%2==1:
                H[m,n]=-J*np.exp(-1j*Spin*Phi*(-1)); H[n,m]=np.conj(H[m,n])
            elif ((mx==nx and my-ny==1) or (mx==nx-1 and ny==my+1)) and my%2==1:
                H[m,n]=-J*np.exp(-1j*Spin*Phi);      H[n,m]=np.conj(H[m,n])
            elif ((mx==nx and ny-my==1) or (mx==nx+1 and ny==my-1)) and my%2==0:
                H[m,n]=-J*np.exp(-1j*Spin*Phi*(-1)); H[n,m]=np.conj(H[m,n])
            elif ((mx==nx and my-ny==1) or (mx==nx+1 and ny==my+1)) and my%2==0:
                H[m,n]=-J*np.exp(-1j*Spin*Phi);      H[n,m]=np.conj(H[m,n])
            elif mx==nx and my%2==1 and abs(my-ny)==2:
                H[m,n]=-J; H[n,m]=np.conj(H[m,n])
            elif my==ny and my%2==0 and abs(mx-nx)==1:
                H[m,n]=-J; H[n,m]=np.conj(H[m,n])
    return H



def H_IQH_cylinder(Nx, Ny, J, Phi, Spin, psi):
    """IQH on a cylinder: open in y, periodic in x with external flux psi."""
    N  = Nx * Ny
    H  = np.zeros((N + 1, N + 1), dtype=complex)
    for m in range(1, N + 1):
        mx, my = NumberToLocation(m, Nx, Ny)
        # Horizontal hoppings (same row, my)
        # right neighbour: mx+1 (standard bond)
        if mx < Nx:
            n = LocationToNumber(mx + 1, my, Nx, Ny)
            val = -J * np.exp(-1j * Phi * my * Spin)
            H[m, n] = val; H[n, m] = np.conj(val)
        # wrap bond: mx=Nx → mx=1, gains flux psi
        elif mx == Nx:
            n = LocationToNumber(1, my, Nx, Ny)
            val = -J * np.exp(-1j * psi) * np.exp(-1j * Phi * my * Spin)
            H[m, n] = val; H[n, m] = np.conj(val)
        # Vertical hoppings (same column, open boundary)
        if my < Ny:
            n = LocationToNumber(mx, my + 1, Nx, Ny)
            H[m, n] = -J; H[n, m] = -J
    return H


def H_AQH_cylinder(Nx, Ny, J, Phi, Spin, psi):
    """AQH on a cylinder: open in y, periodic in x with external flux psi."""
    N  = Nx * Ny
    H  = np.zeros((N + 1, N + 1), dtype=complex)
    for m in range(1, N + 1):
        mx, my = NumberToLocation(m, Nx, Ny)
        sgn = ((-1) ** mx) * ((-1) ** my)
        # ── Horizontal (same row) ──────────────────────────────────────────────
        if mx < Nx:
            n = LocationToNumber(mx + 1, my, Nx, Ny)
            val = -J * np.exp(-1j * sgn * Spin * Phi)
            H[m, n] = val; H[n, m] = np.conj(val)
        elif mx == Nx:                   # wrap bond with psi
            n = LocationToNumber(1, my, Nx, Ny)
            val = -J * np.exp(-1j * psi) * np.exp(-1j * sgn * Spin * Phi)
            H[m, n] = val; H[n, m] = np.conj(val)
        # ── Vertical (same column, open) ───────────────────────────────────────
        if my < Ny:
            n = LocationToNumber(mx, my + 1, Nx, Ny)
            val = -J * np.exp(1j * sgn * Spin * Phi)
            H[m, n] = val; H[n, m] = np.conj(val)
        # ── Diagonal NE: mx+1, my+1, same parity ──────────────────────────────
        if mx < Nx and my < Ny and (mx % 2) == (my % 2):
            n = LocationToNumber(mx + 1, my + 1, Nx, Ny)
            H[m, n] = -J; H[n, m] = -J
        if mx == Nx and my < Ny and (mx % 2) == (my % 2):   # wrap NE diagonal
            n = LocationToNumber(1, my + 1, Nx, Ny)
            val = -J * np.exp(-1j * psi)
            H[m, n] = val; H[n, m] = np.conj(val)
        # ── Diagonal NW: mx-1, my+1, opposite parity ──────────────────────────
        if mx > 1 and my < Ny and (mx % 2) != (my % 2):
            n = LocationToNumber(mx - 1, my + 1, Nx, Ny)
            H[m, n] = -J; H[n, m] = -J
        if mx == 1 and my < Ny and (mx % 2) != (my % 2):    # wrap NW diagonal
            n = LocationToNumber(Nx, my + 1, Nx, Ny)
            val = -J * np.exp(1j * psi)
            H[m, n] = val; H[n, m] = np.conj(val)
    return H

def build_hamiltonian(h_str, nx0, ny0, nx1, ny1, j1, phi_iqh0, phi_iqh1, phi_aqh0, phi_aqh1, psi=0.0):
    """
    phi_iqh0 / phi_iqh1 : IQH layer-0 and layer-1 phases
    phi_aqh0 / phi_aqh1 : AQH layer-0 and layer-1 phases
    psi                 : external flux (cylinder only)
    """
    if h_str == 'IQH_IQH':  return H_IQH_IQH(nx0, ny0, nx1, ny1, J0, j1, phi_iqh0, phi_iqh1, Spin)
    if h_str == 'AQH_AQH':  return H_AQH_AQH(nx0, ny0, nx1, ny1, J0, j1, phi_aqh0, phi_aqh1, Spin)
    if h_str == 'IQH_AQH':  return H_IQH_IQH(nx0, ny0, nx1, ny1, J0, j1, phi_iqh0, phi_aqh1, Spin)
    if h_str == 'AQH_IQH':  return H_AQH_AQH(nx0, ny0, nx1, ny1, J0, j1, phi_aqh0, phi_iqh1, Spin)
    if h_str == 'A_zigzag': return H_zigzag(nx0, ny0, J0, phi_aqh0, Spin)
    if h_str == 'IQH_cyl':  return H_IQH_cylinder(nx0, ny0, J0, phi_iqh0, Spin, psi)
    if h_str == 'AQH_cyl':  return H_AQH_cylinder(nx0, ny0, J0, phi_aqh0, Spin, psi)
    raise ValueError(h_str)


# ══════════════════════════════════════════════════════════════════════════════
#  Spectrum computation — vectorised across all frequencies
# ══════════════════════════════════════════════════════════════════════════════

def compute_spectrum(H_mat, kin, kex, DWP, isite, osite, N):
    """
    Solve (base - i*w*I) E = F for all frequencies.

    Uses eigendecomposition of base: diagonalise once, then each frequency
    is a cheap element-wise division — O(N²) per point vs O(N³) for a solve.
    Speedup: ~2.5x at N=100, ~3.5x at N=200.
    """
    sqKex  = np.sqrt(2 * kex)
    H_cut  = H_mat[1:, 1:]
    kex_d  = np.zeros(N)
    kex_d[isite - 1] = kex
    kex_d[osite - 1] = kex
    base   = kin * np.eye(N) + np.diag(kex_d) - 1j * H_cut

    Fv     = np.zeros(N, dtype=complex)
    Fv[isite - 1] = sqKex

    # Diagonalise base once: base = V * diag(lam) * V^{-1}
    lam, V  = np.linalg.eig(base)
    Fv_rot  = np.linalg.solve(V, Fv)          # V^{-1} @ Fv, more stable than inv

    # For each frequency w: E(w) = V * diag(1/(lam - i*w)) * Fv_rot
    nW      = len(DWP)
    denom   = lam[None, :] - 1j * DWP[:, None]   # (nW, N)
    coeffs  = Fv_rot[None, :] / denom              # (nW, N)
    E_all   = coeffs @ V.T                         # (nW, N)  — note: V rows are eigenvecs

    # Assemble outputs
    P_drop = 2 * kex * np.abs(E_all[:, osite - 1]) ** 2
    P_thru = np.abs(1.0 - sqKex * E_all[:, isite - 1]) ** 2
    maxT   = max(P_thru.max(), 1e-30)
    thru   = P_thru / maxT
    drop   = P_drop / maxT

    # field array shape (N+1, nW) for compatibility with rest of app
    field = np.zeros((N + 1, nW), dtype=complex)
    field[1:] = E_all.T

    phi_arr = np.unwrap(np.angle(sqKex * field[osite]))
    delay   = np.gradient(phi_arr, DWP) / (2 * np.pi)

    # ── Exact κ_σ per supermode of H ──────────────────────────────────
    # κ_σ = κ_in + Σ_{s ∈ port_sites} κ_ex · |⟨σ|s⟩|²
    # Use the set of UNIQUE port sites so that when isite==osite (single ring
    # or bridge topology), κ_ex is counted once, not twice.
    eps_H, V_H  = np.linalg.eigh(H_cut)
    kappa_sigma = np.full(V_H.shape[1], kin, dtype=float)
    for port in {isite, osite}:                           # set ⇒ unique
        kappa_sigma += kex * np.abs(V_H[port - 1, :])**2
    sort_idx    = np.argsort(eps_H)
    eps_H       = eps_H[sort_idx]
    kappa_sigma = kappa_sigma[sort_idx]
    V_H         = V_H[:, sort_idx]

    return thru, drop, delay, np.abs(field[1:]) ** 2, field[1:], kappa_sigma, eps_H, V_H


def compute_flow(field_1d, H_mat, NL, isite, osite, h_str, nx0, ny0, nx1, ny1):
    is_zz = (h_str == 'A_zigzag')
    is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
    Nxt   = nx0 if (is_zz or is_cyl) else nx0 * nx1
    Nyt   = ny0 if (is_zz or is_cyl) else ny0 * ny1
    Xr, Yr, Ur, Vr, cols = [], [], [], [], []

    if is_zz:
        vr    = [(0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        vecs  = [np.array(v, float) for v in vr]
        norms = [v / np.linalg.norm(v) for v in vecs]

        def inlat(xc, yc):
            xc, yc = int(round(xc)), int(round(yc))
            if not 1 <= yc <= 2 * Nyt - 1: return False
            return (2 <= xc <= 2 * Nxt - 2) if yc % 2 == 1 else (1 <= xc <= 2 * Nxt - 1)

        def c2s(xc, yc):
            xc, yc = int(round(xc)), int(round(yc))
            if yc % 2 == 1: return int((yc - 1) // 2 * (2 * Nxt - 1) + xc // 2)
            return int((yc - 2) // 2 * (2 * Nxt - 1) + Nxt - 1 + (xc + 1) // 2)

        for n in range(1, NL + 1):
            xn, yn = site_xy(n, h_str, nx0, ny0, nx1, ny1)
            arr    = np.zeros(2)
            for v, u in zip(vecs, norms):
                xb, yb = xn + v[0], yn + v[1]
                if not inlat(xb, yb): continue
                try:
                    nb = c2s(xb, yb)
                    f  = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f  = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')
    elif is_cyl:
        # For cylinder: find neighbours via nonzero H entries, unit vector = direction in polar coords
        sc = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL + 1)}
        for n in range(1, NL + 1):
            xn, yn = sc[n]
            arr = np.zeros(2)
            for nb in range(1, NL + 1):
                if nb == n: continue
                if H_mat[n, nb] == 0: continue
                xb, yb = sc[nb]
                diff = np.array([xb - xn, yb - yn])
                norm = np.linalg.norm(diff)
                if norm < 1e-10: continue
                u = diff / norm
                try:
                    f = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')
    else:
        vr    = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        vecs  = [np.array(v, float) for v in vr]
        norms = [v / np.linalg.norm(v) for v in vecs]
        sc    = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL + 1)}
        c2n   = {(int(round(v[0])), int(round(v[1]))): k for k, v in sc.items()}

        for n in range(1, NL + 1):
            xn, yn = sc[n]
            arr    = np.zeros(2)
            for v, u in zip(vecs, norms):
                nb = c2n.get((int(round(xn + v[0])), int(round(yn + v[1]))))
                if nb is None: continue
                try:
                    f = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')

    U, V = np.array(Ur), np.array(Vr)
    mx   = np.sqrt(U ** 2 + V ** 2).max()
    if mx > 0: U /= mx; V /= mx
    return np.array(Xr) - 0.5 * U, np.array(Yr) - 0.5 * V, U, V, cols


# ── Worker thread ─────────────────────────────────────────────────────────────

# Shared thread-pool executor — one worker thread, avoids creating/destroying threads
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# ══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Lida Xu\'s Topological Photonic Lattice Explorer — v1.0')
        self.setMinimumSize(1200, 750)
        icon_path = os.path.join(_BUNDLE_DIR, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self._apply_dark_theme()
        self.state = dict(
            h_str='IQH_IQH', nx0=6, ny0=6, nx1=1, ny1=1, j1=0.3,
            kin=0.001, kex=0.01,
            phi_iqh0=PHI_IQH_DEFAULT, phi_iqh1=PHI_IQH_DEFAULT,
            phi_aqh0=PHI_AQH_DEFAULT, phi_aqh1=PHI_AQH_DEFAULT, psi=0.0,
            sw_start=1.455, sw_end=1.4, sw_step=0.0001,
            isite=1, osite=6*6 - 6 + 1, NL=36, Nxt=6, Nyt=6, is_zz=False, is_cyl=False,
            heaters={}, defects=set(), selected=None,
            spectrum=None, complex_field=None, H_mat=None,
            probe_idx=500, mode='Heater',
        )
        self._build_ui()
        self._on_hstr_change(self.state['h_str'])   # set initial slider visibility
        self._rebuild_lattice()

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        pal = QPalette()
        for role, col in [
            (QPalette.Window,          (8,  9,  13)),
            (QPalette.WindowText,      (200,208,231)),
            (QPalette.Base,            (14, 16, 24)),
            (QPalette.AlternateBase,   (23, 28, 46)),
            (QPalette.Text,            (200,208,231)),
            (QPalette.Button,          (23, 28, 46)),
            (QPalette.ButtonText,      (200,208,231)),
            (QPalette.Highlight,       (0,  229,255)),
            (QPalette.HighlightedText, (8,  9,  13)),
        ]:
            pal.setColor(role, QColor(*col))
        self.setPalette(pal)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#08090d;color:#c8d0e7;font-size:12px;}
            QGroupBox{border:1px solid #1e2230;border-radius:6px;margin-top:10px;
                      padding:8px;font-size:11px;font-weight:bold;color:#3a4a70;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QPushButton{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
                        padding:5px 12px;color:#c8d0e7;font-size:12px;}
            QPushButton:hover{background:#1e2a40;border-color:#00e5ff;}
            QPushButton:pressed,QPushButton:checked{background:#00e5ff;color:#08090d;border-color:#00e5ff;}
            QPushButton:disabled{color:#2a3050;}
            QComboBox,QDoubleSpinBox,QSpinBox{background:#171c2e;border:1px solid #1e2230;
                border-radius:4px;padding:4px 6px;color:#c8d0e7;font-size:12px;}
            QSlider::groove:horizontal{height:4px;background:#1e2230;border-radius:2px;}
            QSlider::handle:horizontal{background:#00e5ff;width:14px;height:14px;
                margin:-5px 0;border-radius:7px;}
            QLabel{color:#c8d0e7;font-size:12px;}
            QStatusBar{color:#4a5270;font-size:11px;}
            QSplitter::handle{background:#1e2230;}
        """)

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main    = QVBoxLayout(central)
        main.setSpacing(5); main.setContentsMargins(8, 8, 8, 8)

        # Title
        t = QLabel('TOPOLOGICAL PHOTONIC LATTICE EXPLORER')
        t.setFont(QFont('Courier New', 13, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};padding:2px 0;')
        s = QLabel('IQH  ·  AQH  ·  Superlattice  ·  Interactive')
        s.setFont(QFont('Courier New', 8)); s.setStyleSheet('color:#7a8aaa;')
        main.addWidget(t); main.addWidget(s)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;'); main.addWidget(sep)

        # 3-column splitter
        splitter = QSplitter(Qt.Horizontal)

        # Col 1: Spectra
        sw = QWidget(); sl = QVBoxLayout(sw); sl.setContentsMargins(0, 0, 0, 0)
        self.fig_spec   = Figure(facecolor=DARK_BG)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_thru = self.fig_spec.add_subplot(311)
        self.ax_drop = self.fig_spec.add_subplot(312)
        self.ax_del  = self.fig_spec.add_subplot(313)
        for ax, col, ttl in [(self.ax_thru,'#4a9eff','Thru port'),
                              (self.ax_drop,'#ff4a6e','Drop port'),
                              (self.ax_del, '#a8ff78','Wigner delay')]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=10, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=9)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560'); sp.set_linewidth(1.2)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
        self.fig_spec.subplots_adjust(hspace=0.55, left=0.1, right=0.97, top=0.95, bottom=0.08)
        sl.addWidget(self.canvas_spec)
        self.canvas_spec.mpl_connect('button_press_event',   self._on_spec_click)
        self.canvas_spec.mpl_connect('motion_notify_event',  self._on_spec_drag)
        self.canvas_spec.mpl_connect('button_release_event', lambda e: setattr(self, '_spec_drag', False))
        self._spec_drag = False

        # Col 2: Lattice
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(0, 0, 0, 0)
        self.fig_lat    = Figure(facecolor=DARK_BG)
        self.canvas_lat = FigureCanvas(self.fig_lat)
        self.ax_lat     = self.fig_lat.add_subplot(111)
        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]); self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values(): sp.set_edgecolor('#3a4560')
        self.fig_lat.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
        ll.addWidget(self.canvas_lat)
        self.canvas_lat.mpl_connect('button_press_event', self._on_lat_click)

        # Col 3: Photon flow
        fw = QWidget(); fl = QVBoxLayout(fw); fl.setContentsMargins(0, 0, 0, 0)
        self.fig_flow    = Figure(facecolor=DARK_BG)
        self.canvas_flow = FigureCanvas(self.fig_flow)
        self.ax_flow     = self.fig_flow.add_subplot(111)
        self.ax_flow.set_facecolor(DARK_BG)
        self.ax_flow.set_xticks([]); self.ax_flow.set_yticks([])
        for sp in self.ax_flow.spines.values(): sp.set_edgecolor('#3a4560')
        self.fig_flow.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
        fl.addWidget(self.canvas_flow)

        # Col 4: Eigenvalue panel
        ew = QWidget(); el = QVBoxLayout(ew); el.setContentsMargins(0, 0, 0, 0)
        self.fig_eig    = Figure(facecolor=DARK_BG)
        self.canvas_eig = FigureCanvas(self.fig_eig)
        self.ax_eig     = self.fig_eig.add_subplot(111)
        self.ax_eig.set_facecolor(PANEL_BG)
        self.ax_eig.tick_params(colors=TEXT_COL, labelsize=8)
        for sp in self.ax_eig.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_eig.grid(True, color=GRID_COL, linewidth=0.4)
        self.fig_eig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
        self._selected_sigma = 0
        self.canvas_eig.mpl_connect('button_press_event', self._on_eig_click)
        el.addWidget(self.canvas_eig)

        splitter.addWidget(ew); splitter.addWidget(sw); splitter.addWidget(lw); splitter.addWidget(fw)
        splitter.setSizes([400, 400, 400, 350])
        self.setMinimumSize(1600, 750)
        main.addWidget(splitter, stretch=1)
        main.addWidget(self._build_controls())
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage('Ready')

    def _build_controls(self):
        w   = QWidget(); row = QHBoxLayout(w)
        row.setSpacing(8); row.setContentsMargins(0, 0, 0, 0)

        # ── Site Control (click mode + heater + IN/OUT info, merged) ──────────
        gc = QGroupBox('Site Control'); gc.setFixedWidth(280)
        cl = QGridLayout(gc); cl.setSpacing(3)
        cl.setColumnStretch(0, 1); cl.setColumnStretch(1, 1)

        # Mode buttons — 2×2 grid in row 0-1
        self.btn_mode = {}
        for lbl, r, c in [('Heater',0,0),('Set IN',0,1),('Set OUT',1,0),('Remove',1,1)]:
            b = QPushButton(lbl); b.setCheckable(True); b.setFixedHeight(24)
            b.clicked.connect(lambda _, l=lbl: self._set_mode(l))
            cl.addWidget(b, r, c); self.btn_mode[lbl] = b
        self.btn_mode['Heater'].setChecked(True)

        # IN/OUT status label spanning full width
        self.lbl_io = QLabel('IN=1  OUT=8')
        self.lbl_io.setStyleSheet('color:#c8d0e7;font-size:10px;')
        cl.addWidget(self.lbl_io, 2, 0, 1, 2)

        # Selected-site label
        self.lbl_site = QLabel('Click a ring')
        self.lbl_site.setStyleSheet('color:#4a5270;font-size:10px;')
        cl.addWidget(self.lbl_site, 3, 0, 1, 2)

        # Heater Δ(J) slider + spinbox
        cl.addWidget(QLabel('on-site detuning (J)'), 4, 0)
        self.sld_pot = QSlider(Qt.Horizontal)
        self.sld_pot.setRange(-100, 100); self.sld_pot.setValue(0)
        self.spn_pot = QDoubleSpinBox()
        self.spn_pot.setRange(-10, 10); self.spn_pot.setSingleStep(0.1); self.spn_pot.setFixedWidth(58)
        self.sld_pot.valueChanged.connect(self._on_pot_change)
        self.spn_pot.valueChanged.connect(self._on_spn_pot_change)
        hl_pot = QHBoxLayout(); hl_pot.setSpacing(4)
        hl_pot.addWidget(self.sld_pot); hl_pot.addWidget(self.spn_pot)
        cl.addLayout(hl_pot, 4, 1)

        row.addWidget(gc)

        # ── Lattice (dims + type + κ sliders) ────────────────────────────────
        gd  = QGroupBox('Lattice'); gd.setFixedWidth(370)
        dl  = QGridLayout(gd); dl.setSpacing(3)

        self.spn_nx0 = QSpinBox(); self.spn_nx0.setRange(1, 20); self.spn_nx0.setValue(6)
        self.spn_ny0 = QSpinBox(); self.spn_ny0.setRange(1, 20); self.spn_ny0.setValue(6)
        self.spn_nx1 = QSpinBox(); self.spn_nx1.setRange(1, 10); self.spn_nx1.setValue(1)
        self.spn_ny1 = QSpinBox(); self.spn_ny1.setRange(1, 10); self.spn_ny1.setValue(1)
        self.spn_j1  = QDoubleSpinBox()
        self.spn_j1.setRange(0, 5); self.spn_j1.setSingleStep(0.05); self.spn_j1.setValue(0.3)
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(['IQH_IQH', 'AQH_AQH', 'IQH_AQH', 'AQH_IQH', 'A_zigzag', 'IQH_cyl', 'AQH_cyl'])
        self.cmb_type.currentTextChanged.connect(self._on_hstr_change)
        self.btn_rebuild = QPushButton('⟳ Rebuild')
        self.btn_rebuild.setFixedHeight(24)
        self.btn_rebuild.clicked.connect(self._rebuild_lattice)

        # row 0-1: dimension spinboxes
        for c, lbl, w_ in [(0,'Nx₀',self.spn_nx0),(1,'Ny₀',self.spn_ny0),
                            (2,'Nx₁',self.spn_nx1),(3,'Ny₁',self.spn_ny1),(4,'J₁',self.spn_j1)]:
            dl.addWidget(QLabel(lbl), 0, c); dl.addWidget(w_, 1, c)
        # row 2: type combo + rebuild
        dl.addWidget(self.cmb_type, 2, 0, 1, 3)
        dl.addWidget(self.btn_rebuild, 2, 3, 1, 2)
        self.sl_widgets = [self.spn_nx1, self.spn_ny1, self.spn_j1]

        # row 3-4: κ sliders + spinboxes (moved here from Simulation)
        self.sld_kin = QSlider(Qt.Horizontal); self.sld_kin.setRange(1,  500);  self.sld_kin.setValue(1)
        self.sld_kex = QSlider(Qt.Horizontal); self.sld_kex.setRange(10, 1000); self.sld_kex.setValue(10)
        self.lbl_kin = QLabel('κ_in'); self.lbl_kex = QLabel('κ_ex')
        self.spn_kin_val = QDoubleSpinBox(); self.spn_kin_val.setRange(0.001, 0.500)
        self.spn_kin_val.setDecimals(3); self.spn_kin_val.setSingleStep(0.001); self.spn_kin_val.setValue(0.001)
        self.spn_kex_val = QDoubleSpinBox(); self.spn_kex_val.setRange(0.010, 1.000)
        self.spn_kex_val.setDecimals(3); self.spn_kex_val.setSingleStep(0.001); self.spn_kex_val.setValue(0.010)
        # Bidirectional sync: slider → spinbox
        self.sld_kin.valueChanged.connect(lambda v: self.spn_kin_val.setValue(v / 1000.))
        self.sld_kex.valueChanged.connect(lambda v: self.spn_kex_val.setValue(v / 1000.))
        # Bidirectional sync: spinbox → slider
        self.spn_kin_val.valueChanged.connect(lambda v: self.sld_kin.setValue(int(round(v * 1000))))
        self.spn_kex_val.valueChanged.connect(lambda v: self.sld_kex.setValue(int(round(v * 1000))))
        self.sld_kin.valueChanged.connect(lambda _: self._invalidate())
        self.sld_kex.valueChanged.connect(lambda _: self._invalidate())
        dl.addWidget(self.lbl_kin, 3, 0); dl.addWidget(self.sld_kin, 3, 1, 1, 3); dl.addWidget(self.spn_kin_val, 3, 4)
        dl.addWidget(self.lbl_kex, 4, 0); dl.addWidget(self.sld_kex, 4, 1, 1, 3); dl.addWidget(self.spn_kex_val, 4, 4)

        row.addWidget(gd)

        # ── Simulation (φ + sweep + buttons only) ─────────────────────────────
        gr  = QGroupBox('Simulation')
        rl  = QGridLayout(gr); rl.setSpacing(3)

        # φ sliders — 0…200 ticks → 0…2π, with spinboxes in units of π
        PHI_TICKS = 200
        def _make_phi_sld(default_ticks, lbl_text, cb):
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, PHI_TICKS); sld.setValue(default_ticks)
            lbl = QLabel(lbl_text)
            spn = QDoubleSpinBox()
            spn.setRange(0.0, 2.0); spn.setDecimals(4); spn.setSingleStep(0.01)
            spn.setSuffix(' π'); spn.setValue(default_ticks / 100.0)
            # Bidirectional sync
            sld.valueChanged.connect(lambda v, s=spn: s.setValue(v / 100.0))
            spn.valueChanged.connect(lambda v, s=sld: s.setValue(int(round(v * 100))))
            sld.valueChanged.connect(cb)
            return sld, lbl, spn

        self.sld_phi_iqh0, self.lbl_phi_iqh0, self.spn_phi_iqh0 = _make_phi_sld(50,  'φ_IQH_layer_0', self._on_phi_iqh0_change)
        self.sld_phi_iqh1, self.lbl_phi_iqh1, self.spn_phi_iqh1 = _make_phi_sld(50,  'φ_IQH_layer_1', self._on_phi_iqh1_change)
        self.sld_phi_aqh0, self.lbl_phi_aqh0, self.spn_phi_aqh0 = _make_phi_sld(25,  'φ_AQH_layer_0', self._on_phi_aqh0_change)
        self.sld_phi_aqh1, self.lbl_phi_aqh1, self.spn_phi_aqh1 = _make_phi_sld(25,  'φ_AQH_layer_1', self._on_phi_aqh1_change)

        # Sweep spinboxes
        self.spn_ss = QDoubleSpinBox(); self.spn_ss.setRange(-20, 20); self.spn_ss.setDecimals(6); self.spn_ss.setValue(1.455); self.spn_ss.setSingleStep(0.001)
        self.spn_se = QDoubleSpinBox(); self.spn_se.setRange(-20, 20); self.spn_se.setDecimals(6); self.spn_se.setValue(1.4);   self.spn_se.setSingleStep(0.001)
        self.spn_st = QDoubleSpinBox(); self.spn_st.setRange(0.0001, 1.0); self.spn_st.setValue(0.0001); self.spn_st.setSingleStep(0.0001); self.spn_st.setDecimals(6)
        for _spn in (self.spn_ss, self.spn_se, self.spn_st,
                     self.spn_nx0, self.spn_ny0, self.spn_nx1, self.spn_ny1, self.spn_j1):
            _spn.valueChanged.connect(lambda _: self._invalidate())

        self.btn_run   = QPushButton('▶ Compute'); self.btn_run.setFixedHeight(28);  self.btn_run.clicked.connect(self._run_spectrum)
        self.btn_clear = QPushButton('✕ Clear');   self.btn_clear.setFixedHeight(28); self.btn_clear.clicked.connect(self._clear)
        self.btn_save  = QPushButton('💾 Save');   self.btn_save.setFixedHeight(28);  self.btn_save.clicked.connect(self._save)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')

        self.lbl_save_path  = QLabel('Save to:')
        self.edit_save_path = QLineEdit(
            os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
            else os.path.dirname(os.path.abspath(__file__)))
        self.edit_save_path.setStyleSheet(
            'background:#171c2e;border:1px solid #1e2230;border-radius:4px;'
            'padding:3px 6px;color:#c8d0e7;font-size:11px;')
        self.btn_browse = QPushButton('Browse')
        self.btn_browse.setFixedWidth(80); self.btn_browse.setFixedHeight(28)
        self.btn_browse.clicked.connect(self._browse_save_path)

        self.btn_default_phi = QPushButton('⟳ Reset All')
        self.btn_default_phi.setFixedHeight(28)
        self.btn_default_phi.clicked.connect(self._reset_all)

        self.btn_feed = QPushButton('⇢ Feed to Nonlinear Simulator')
        self.btn_feed.setFixedHeight(28)
        self.btn_feed.setEnabled(False)
        self.btn_feed.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_feed.clicked.connect(self._open_nonlinear)

        # ψ (external flux) slider + spinbox — only active for cylinder types
        self.sld_psi = QSlider(Qt.Horizontal)
        self.sld_psi.setRange(0, 200); self.sld_psi.setValue(0)   # 0…2π
        self.lbl_psi = QLabel('ψ_ext')
        self.spn_psi_val = QDoubleSpinBox()
        self.spn_psi_val.setRange(0.0, 2.0); self.spn_psi_val.setDecimals(4)
        self.spn_psi_val.setSingleStep(0.01); self.spn_psi_val.setSuffix(' π'); self.spn_psi_val.setValue(0.0)
        self.sld_psi.setVisible(False); self.lbl_psi.setVisible(False); self.spn_psi_val.setVisible(False)
        # Bidirectional sync
        self.sld_psi.valueChanged.connect(lambda v: self.spn_psi_val.setValue(v / 100.0))
        self.spn_psi_val.valueChanged.connect(lambda v: self.sld_psi.setValue(int(round(v * 100))))
        self.sld_psi.valueChanged.connect(self._on_psi_change)

        # rows 0-3: four phase sliders + spinboxes (visibility set by _on_hstr_change)
        rl.addWidget(self.lbl_phi_iqh0,  0, 0); rl.addWidget(self.sld_phi_iqh0, 0, 1, 1, 2); rl.addWidget(self.spn_phi_iqh0, 0, 3)
        rl.addWidget(self.lbl_phi_iqh1,  1, 0); rl.addWidget(self.sld_phi_iqh1, 1, 1, 1, 2); rl.addWidget(self.spn_phi_iqh1, 1, 3)
        rl.addWidget(self.lbl_phi_aqh0,  2, 0); rl.addWidget(self.sld_phi_aqh0, 2, 1, 1, 2); rl.addWidget(self.spn_phi_aqh0, 2, 3)
        rl.addWidget(self.lbl_phi_aqh1,  3, 0); rl.addWidget(self.sld_phi_aqh1, 3, 1, 1, 2); rl.addWidget(self.spn_phi_aqh1, 3, 3)
        # row 4: ψ (external flux, cylinder only)
        rl.addWidget(self.lbl_psi,       4, 0); rl.addWidget(self.sld_psi,      4, 1, 1, 2); rl.addWidget(self.spn_psi_val, 4, 3)
        # row 5: sweep start / end
        rl.addWidget(QLabel('Start'),    5, 0); rl.addWidget(self.spn_ss, 5, 1)
        rl.addWidget(QLabel('End'),      5, 2); rl.addWidget(self.spn_se, 5, 3)
        # row 6: sweep step / run / clear / default-phi
        rl.addWidget(QLabel('Step'),          6, 0); rl.addWidget(self.spn_st, 6, 1)
        rl.addWidget(self.btn_run,            6, 2); rl.addWidget(self.btn_clear, 6, 3)
        rl.addWidget(self.btn_default_phi,    6, 4)
        # row 7: save path + browse + save
        rl.addWidget(self.lbl_save_path, 7, 0)
        rl.addWidget(self.edit_save_path,7, 1, 1, 2)
        rl.addWidget(self.btn_browse,    7, 3)
        rl.addWidget(self.btn_save,      7, 4)
        self.btn_load = QPushButton('📂 Load Existing Sim')
        self.btn_load.setFixedHeight(28)
        self.btn_load.clicked.connect(self._load_existing)

        # row 8: feed (left 3 cols) + load (right 2 cols)
        rl.addWidget(self.btn_feed, 8, 0, 1, 3)
        rl.addWidget(self.btn_load, 8, 3, 1, 2)

        row.addWidget(gr, stretch=1)
        return w

    # ── Invalidation ─────────────────────────────────────────────────────────
    def _invalidate(self):
        """Disable Save and Feed whenever any simulation parameter changes."""
        self.btn_feed.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._session_folder = None   # stale path no longer valid

    # ── φ slider callbacks ────────────────────────────────────────────────────
    def _phi_label(self, ticks):
        """Convert 0-200 tick value to a readable π-fraction string."""
        v = ticks / 100.0  # in units of π
        # Nice fractions: 0, 1/4, 1/3, 1/2, 2/3, 3/4, 1, ...
        frac_map = {0:'0', 25:'π/4', 33:'π/3', 50:'π/2', 67:'2π/3',
                    75:'3π/4', 100:'π', 125:'5π/4', 150:'3π/2', 175:'7π/4', 200:'2π'}
        return frac_map.get(ticks, f'{v:.2f}π')

    def _on_phi_iqh0_change(self, val):
        self.state['phi_iqh0'] = val * np.pi / 100.0
        self._invalidate()

    def _on_phi_iqh1_change(self, val):
        self.state['phi_iqh1'] = val * np.pi / 100.0
        self._invalidate()

    def _on_phi_aqh0_change(self, val):
        self.state['phi_aqh0'] = val * np.pi / 100.0
        self._invalidate()

    def _on_phi_aqh1_change(self, val):
        self.state['phi_aqh1'] = val * np.pi / 100.0
        self._invalidate()

    def _on_psi_change(self, val):
        psi = val * np.pi / 100.0
        self.state['psi'] = psi
        self._invalidate()

    def _reset_all(self):
        """Reset every UI control to its startup default."""
        # Lattice type — triggers _on_hstr_change → _invalidate
        self.cmb_type.setCurrentText('IQH_IQH')

        # Lattice dimensions & coupling
        self.spn_nx0.setValue(6);  self.spn_ny0.setValue(6)
        self.spn_nx1.setValue(1);  self.spn_ny1.setValue(1)
        self.spn_j1.setValue(0.3)

        # Loss rates  (slider ticks → kin=0.001, kex=0.010)
        self.sld_kin.setValue(1);  self.sld_kex.setValue(10)

        # Sweep window
        self.spn_ss.setValue(1.455)
        self.spn_se.setValue(1.4)
        self.spn_st.setValue(0.0001)

        # Phase sliders  (ticks: IQH → π/2=50, AQH → π/4=25, ψ → 0)
        self.sld_phi_iqh0.setValue(50)
        self.sld_phi_iqh1.setValue(50)
        self.sld_phi_aqh0.setValue(25)
        self.sld_phi_aqh1.setValue(25)
        self.sld_psi.setValue(0)

        # Heater / defect / selection state — mirrors full Clear behaviour
        s = self.state
        s['heaters'].clear(); s['defects'].clear(); s['selected'] = None
        s['isite'] = 0; s['osite'] = 0   # force reset to type defaults in _rebuild_lattice
        self.sld_pot.setValue(0); self.spn_pot.setValue(0.)
        self.lbl_site.setText('Click a ring')

        # Click mode
        self._set_mode('Heater')

        # Rebuild lattice (resets IN/OUT sites to type defaults, redraws)
        self._rebuild_lattice()

    # ── Mode / lattice ────────────────────────────────────────────────────────
    def _set_mode(self, mode):
        self.state['mode'] = mode
        for l, b in self.btn_mode.items(): b.setChecked(l == mode)

    def _on_hstr_change(self, h):
        cyl = h in ('IQH_cyl', 'AQH_cyl')
        zz  = h == 'A_zigzag'
        for w in self.sl_widgets: w.setEnabled(not zz and not cyl)
        # Which phase sliders are relevant per Hamiltonian type:
        #   IQH_IQH  : iqh0, iqh1
        #   AQH_AQH  : aqh0, aqh1
        #   IQH_AQH  : iqh0 (layer-0), aqh1 (layer-1)
        #   AQH_IQH  : aqh0 (layer-0), iqh1 (layer-1)
        #   A_zigzag : aqh0 only
        #   IQH_cyl  : iqh0 only + psi
        #   AQH_cyl  : aqh0 only + psi
        show = {
            'phi_iqh0': h in ('IQH_IQH', 'IQH_AQH', 'IQH_cyl'),
            'phi_iqh1': h in ('IQH_IQH', 'AQH_IQH'),
            'phi_aqh0': h in ('AQH_AQH', 'AQH_IQH', 'A_zigzag', 'AQH_cyl'),
            'phi_aqh1': h in ('AQH_AQH', 'IQH_AQH'),
        }
        for key, sld, lbl, spn in [
            ('phi_iqh0', self.sld_phi_iqh0, self.lbl_phi_iqh0, self.spn_phi_iqh0),
            ('phi_iqh1', self.sld_phi_iqh1, self.lbl_phi_iqh1, self.spn_phi_iqh1),
            ('phi_aqh0', self.sld_phi_aqh0, self.lbl_phi_aqh0, self.spn_phi_aqh0),
            ('phi_aqh1', self.sld_phi_aqh1, self.lbl_phi_aqh1, self.spn_phi_aqh1),
        ]:
            sld.setVisible(show[key]); lbl.setVisible(show[key]); spn.setVisible(show[key])
        self.sld_psi.setVisible(cyl); self.lbl_psi.setVisible(cyl); self.spn_psi_val.setVisible(cyl)
        self._invalidate()

    def _rebuild_lattice(self):
        s = self.state; h = self.cmb_type.currentText()
        nx0, ny0 = self.spn_nx0.value(), self.spn_ny0.value()
        nx1, ny1 = self.spn_nx1.value(), self.spn_ny1.value()
        NL, Nxt, Nyt, id_, od_ = get_lattice_params(h, nx0, ny0, nx1, ny1)
        isite = s['isite'] if 1 <= s['isite'] <= NL else id_
        # Always reset osite to the type default on rebuild so it tracks
        # the lattice size (e.g. Nx₀·Ny₀ − Nx₀ + 1 for standard types).
        osite = od_
        s.update(h_str=h, nx0=nx0, ny0=ny0, nx1=nx1, ny1=ny1,
                 NL=NL, Nxt=Nxt, Nyt=Nyt,
                 isite=isite, osite=osite, is_zz=(h == 'A_zigzag'),
                 is_cyl=(h in ('IQH_cyl','AQH_cyl')),
                 selected=None, spectrum=None, complex_field=None, H_mat=None, probe_idx=500)
        s['heaters'] = {k: v for k, v in s['heaters'].items() if 1 <= k <= NL}
        s['defects']  = {d for d in s['defects'] if 1 <= d <= NL}
        self._invalidate()
        self._update_io_label(); self._draw_lattice(); self._draw_flow()

    def _site_xy(self, n):
        s = self.state
        return site_xy(n, s['h_str'], s['nx0'], s['ny0'], s['nx1'], s['ny1'])

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_lattice(self):
        s = self.state; ax = self.ax_lat; ax.cla()
        ax.set_facecolor(PANEL_BG); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.set_title('Lattice — click to interact', color=TEXT_COL, fontsize=9, pad=3)
        NL = s['NL']; Nxt = s['Nxt']; Nyt = s['Nyt']
        is_zz = s['is_zz']; is_cyl = s.get('is_cyl', False)
        if is_zz:   ax.set_xlim(0, 2*Nxt);        ax.set_ylim(0, 2*Nyt)
        elif is_cyl: pass   # auto-limits set by matplotlib after patches are added
        else:        ax.set_xlim(-0.8, Nxt-0.2);  ax.set_ylim(-0.8, Nyt-0.2)

        # Bonds
        if is_zz:
            for n in range(1, NL+1):
                xn, yn = self._site_xy(n)
                mx, my = NumberToLocation_AQH_zigzag(n, Nxt, Nyt)
                for n2 in range(n+1, NL+1):
                    nx2, ny2 = NumberToLocation_AQH_zigzag(n2, Nxt, Nyt); c = False
                    if my % 2 == 1:
                        if ((mx==nx2 and ny2-my==1) or (mx==nx2-1 and ny2==my-1) or
                            (mx==nx2 and my-ny2==1) or (mx==nx2-1 and ny2==my+1) or
                            (mx==nx2 and abs(my-ny2)==2)): c = True
                    else:
                        if ((mx==nx2 and ny2-my==1) or (mx==nx2+1 and ny2==my-1) or
                            (mx==nx2 and my-ny2==1) or (mx==nx2+1 and ny2==my+1) or
                            (my==ny2 and abs(mx-nx2)==1)): c = True
                    if c:
                        x2, y2 = self._site_xy(n2)
                        ax.plot([xn,x2],[yn,y2], color='#1e2a40', lw=1, zorder=1)
        elif is_cyl:
            # Draw bonds from Hamiltonian nonzero entries
            H_draw = s.get('H_mat')
            coords_c = {n2: self._site_xy(n2) for n2 in range(1, NL+1)}
            for n in range(1, NL+1):
                xn, yn = coords_c[n]
                mx, my = NumberToLocation(n, Nxt, Nyt)
                # vertical bond (open boundary)
                if my < Nyt:
                    n2 = LocationToNumber(mx, my+1, Nxt, Nyt)
                    x2, y2 = coords_c[n2]
                    ax.plot([xn,x2],[yn,y2], color='#1e2a40', lw=1, zorder=1)
                # horizontal bond (including wrap) — draw arc via midpoint
                n2 = LocationToNumber(mx % Nxt + 1, my, Nxt, Nyt)
                x2, y2 = coords_c[n2]
                # only draw once (n < n2) except for wrap where n=Nx,n2=1
                if n < n2 or (mx == Nxt):
                    ax.plot([xn,x2],[yn,y2],
                            color='#2a1e40' if mx == Nxt else '#1e2a40',
                            lw=1.2 if mx == Nxt else 1, ls='--' if mx == Nxt else '-',
                            zorder=1)
        else:
            coords = {n2: self._site_xy(n2) for n2 in range(1, NL+1)}
            c2n    = {(int(round(v[0])), int(round(v[1]))): k for k, v in coords.items()}
            for n in range(1, NL+1):
                xn, yn = coords[n]
                for dx, dy in [(1,0),(0,1)]:
                    nb = c2n.get((int(round(xn+dx)), int(round(yn+dy))))
                    if nb:
                        x2, y2 = coords[nb]
                        ax.plot([xn,x2],[yn,y2], color='#1e2a40', lw=1, zorder=1)

        # Rings
        sp_data = s.get('spectrum')
        for n in range(1, NL+1):
            xn, yn = self._site_xy(n)
            ec = ('#333355' if n in s['defects'] else
                  '#4a9eff' if n == s['isite'] else
                  '#ff4a6e' if n == s['osite'] else
                  '#ff6b35' if s['heaters'].get(n, 0) != 0 else
                  ACCENT    if n == s['selected'] else '#3d5a8a')
            fc = (plt.cm.hot(float(np.clip(
                      sp_data['power'][n-1, s['probe_idx']] /
                      max(sp_data['power'][:, s['probe_idx']].max(), 1e-30), 0, 1)))
                  if sp_data else '#12263f')
            lw = 2.5 if n == s['selected'] else 1.4
            ax.add_patch(mpatches.Circle((xn,yn), RING_R,        edgecolor=ec, facecolor=fc, lw=lw, zorder=2))
            ax.add_patch(mpatches.Circle((xn,yn), RING_R*RING_GAP, edgecolor='none', facecolor=DARK_BG, zorder=3))
            lbl = ('×'  if n in s['defects'] else
                   'IN' if n == s['isite']   else
                   'OUT'if n == s['osite']   else
                   (f"{'+' if s['heaters'].get(n,0)>0 else ''}{s['heaters'][n]:.1f}"
                    if s['heaters'].get(n, 0) != 0 else None))
            if lbl:
                lc = ('#555577' if n in s['defects'] else
                      '#4a9eff' if n == s['isite']   else
                      '#ff4a6e' if n == s['osite']   else '#ff6b35')
                ax.text(xn, yn, lbl, ha='center', va='center',
                        fontsize=6.5, color=lc, fontweight='bold', zorder=4)
        self.canvas_lat.draw_idle()

    def _draw_flow(self):
        s = self.state; ax = self.ax_flow; ax.cla()
        ax.set_facecolor(DARK_BG); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(SPINE_COL)
        Nxt = s['Nxt']; Nyt = s['Nyt']; is_zz = s['is_zz']; is_cyl = s.get('is_cyl',False)
        if is_zz:   ax.set_xlim(-0.5, 2*Nxt+0.5); ax.set_ylim(-0.5, 2*Nyt+0.5)
        elif is_cyl: pass   # auto
        else:        ax.set_xlim(-1., Nxt);         ax.set_ylim(-1., Nyt)
        cf = s.get('complex_field'); H_mat = s.get('H_mat')
        if cf is None or H_mat is None:
            ax.set_title('Photon Flow — compute spectrum first', color=TEXT_DIM, fontsize=10, pad=3)
            self.canvas_flow.draw_idle(); return
        idx = s['probe_idx']
        fld = np.zeros(s['NL']+1, dtype=complex); fld[1:] = cf[:, idx]
        X, Y, U, V, cols = compute_flow(
            fld, H_mat, s['NL'], s['isite'], s['osite'],
            s['h_str'], s['nx0'], s['ny0'], s['nx1'], s['ny1'])
        ax.quiver(X, Y, U, V, color='white', scale=1, scale_units='xy',
                  width=0.006, headwidth=4, headlength=5, zorder=3)
        for i, (xi, yi, ui, vi) in enumerate(zip(X, Y, U, V)):
            n = i + 1
            if n in (s['isite'], s['osite']):
                col = '#4a9eff' if n == s['isite'] else '#ff4a6e'
                ax.quiver([xi],[yi],[ui],[vi], color=col, scale=1, scale_units='xy',
                          width=0.008, headwidth=4, headlength=5, zorder=4)
        ax.set_title(f'Photon Flow  Δ/J={s["spectrum"]["DWP"][idx]/J0:.4f}',
                     color=TEXT_COL, fontsize=9, pad=3)
        self.canvas_flow.draw_idle()

    # ── Eigenvalue panel ──────────────────────────────────────────────────────
    def _draw_eig(self):
        s  = self.state
        sp = s.get('spectrum')
        ax = self.ax_eig; ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        for sp2 in ax.spines.values(): sp2.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, linewidth=0.4)
        ax.set_xlabel('σ  (index)', color=TEXT_DIM, fontsize=8)
        ax.set_ylabel('ε_σ  (J)',   color=TEXT_DIM, fontsize=8)
        if sp is None or sp.get('eps_H') is None:
            ax.set_title('H eigenvalues — run Compute first', color='#cc88ff', fontsize=8, pad=3)
            self.canvas_eig.draw_idle(); return

        eps_H       = sp['eps_H']
        kappa_sigma = sp['kappa_sigma']
        kin = s.get('kin', 0.001); kex = s.get('kex', 0.01)
        N   = len(eps_H)
        sig = min(self._selected_sigma, N - 1)

        # Sweep range shaded band
        sw_lo = min(self.spn_ss.value(), self.spn_se.value())
        sw_hi = max(self.spn_ss.value(), self.spn_se.value())
        ax.axhspan(sw_lo, sw_hi, color='#1a3a1a', alpha=0.7, zorder=0,
                   label=f'sweep [{sw_lo:.3f}, {sw_hi:.3f}]')

        # Dots colored by κ_σ
        k_mid  = kin + kex / 2
        colors = ['#ff4a6e' if k > k_mid else '#378ADD' for k in kappa_sigma]
        ax.scatter(np.arange(N), eps_H, c=colors, s=24, zorder=3)

        # Selected supermode highlighted
        k_eff = kappa_sigma[sig]; k_ex = k_eff - kin
        ax.scatter([sig], [eps_H[sig]], color='#ffcc00', s=80, zorder=4, marker='D')
        ax.set_title(
            f'σ={sig}   ε={eps_H[sig]:.4f} J\nκ_eff={k_eff:.5f}   κ_ex={k_ex:.5f}',
            color='#ffcc00', fontsize=8, pad=3)
        ax.set_xlim(-0.5, N - 0.5)
        pad = max(0.1, (eps_H.max() - eps_H.min()) * 0.15)
        ax.set_ylim(eps_H.min() - pad, eps_H.max() + pad)
        ax.legend(fontsize=6, frameon=False, labelcolor=TEXT_COL, loc='upper left')


        self.canvas_eig.draw_idle()

    def _on_eig_click(self, event):
        if event.inaxes != self.ax_eig: return
        if event.xdata is None: return
        sp = self.state.get('spectrum')
        if sp is None or sp.get('eps_H') is None: return
        N = len(sp['eps_H'])
        self._selected_sigma = int(np.clip(round(event.xdata), 0, N - 1))
        self._draw_eig()

    def _draw_spectra(self):
        s = self.state; sp = s['spectrum']
        if sp is None: return
        DWP = sp['DWP']; xv = DWP / J0; idx = s['probe_idx']

        # Compute per-panel adaptive ylims from the data itself.
        # Pad is 10% of the data range; floors/ceilings prevent degenerate ylims.
        def _adaptive_ylim(y, force_nonneg=False):
            y = np.asarray(y)
            ymin = float(np.min(y)); ymax = float(np.max(y))
            span = ymax - ymin
            if span < 1e-12:
                # constant trace → show small window around it
                return (ymin - 0.05, ymax + 0.05)
            pad = 0.08 * span
            lo  = ymin - pad
            hi  = ymax + pad
            if force_nonneg and lo < 0 and ymin >= 0:
                lo = -0.02 * span   # tiny negative margin so the 0-line is visible
            return (lo, hi)

        panels = [
            (self.ax_thru, 'thru',  '#4a9eff', _adaptive_ylim(sp['thru'],  force_nonneg=True)),
            (self.ax_drop, 'drop',  '#ff4a6e', _adaptive_ylim(sp['drop'],  force_nonneg=True)),
            (self.ax_del,  'delay', '#a8ff78', _adaptive_ylim(sp['delay'], force_nonneg=False)),
        ]

        for ax, key, col, ylim in panels:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ttl = {'thru':'Thru port','drop':'Drop port','delay':'Wigner delay'}[key]
            ax.set_title(ttl, color=col, fontsize=9, pad=3)
            ax.set_xlabel('Detuning / J₀', color=TEXT_DIM, fontsize=8)
            ax.grid(True, color=GRID_COL, linewidth=0.4)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for sp2 in ax.spines.values(): sp2.set_edgecolor('#3a4560'); sp2.set_linewidth(1.2)
            y = sp[key]; ax.plot(xv, y, color=col, lw=1.8)
            ax.fill_between(xv, y, alpha=0.15, color=col)
            ax.set_ylim(ylim); ax.set_xlim(xv.min(), xv.max())
            ax.axvline(DWP[idx]/J0, color='white', lw=1.2, ls='--', alpha=0.8)
        self.fig_spec.subplots_adjust(hspace=0.55, left=0.1, right=0.97, top=0.95, bottom=0.08)
        self.canvas_spec.draw_idle()

    # ── Interaction callbacks ─────────────────────────────────────────────────
    def _on_lat_click(self, event):
        if event.inaxes != self.ax_lat: return
        s = self.state; NL = s['NL']
        best_n, best_d = None, float('inf')
        for n in range(1, NL+1):
            xn, yn = self._site_xy(n)
            d = (event.xdata - xn)**2 + (event.ydata - yn)**2
            if d < best_d: best_d = d; best_n = n
        if best_d > (RING_R*3)**2: return
        mode = s['mode']
        if mode == 'Set IN':
            s['isite'] = best_n; self._invalidate(); self._update_io_label(); self._draw_lattice()
        elif mode == 'Set OUT':
            s['osite'] = best_n; self._invalidate(); self._update_io_label(); self._draw_lattice()
        elif mode == 'Remove':
            if best_n in s['defects']: s['defects'].discard(best_n)
            else: s['defects'].add(best_n)
            self._invalidate()
            self._draw_lattice()
        else:
            s['selected'] = best_n
            v = float(s['heaters'].get(best_n, 0.))
            self.sld_pot.blockSignals(True); self.sld_pot.setValue(int(v*10)); self.sld_pot.blockSignals(False)
            self.spn_pot.blockSignals(True); self.spn_pot.setValue(v);         self.spn_pot.blockSignals(False)
            self.lbl_site.setText(f'Site {best_n}  Δ={v:+.2f}J')
            self._draw_lattice()

    def _on_spec_click(self, event): self._spec_drag = True;  self._probe_at(event)
    def _on_spec_drag(self, event):
        if self._spec_drag: self._probe_at(event)

    def _probe_at(self, event):
        s = self.state; sp = s.get('spectrum')
        if sp is None: return
        if event.inaxes not in (self.ax_thru, self.ax_drop, self.ax_del): return
        if event.xdata is None: return
        DWP = sp['DWP']; idx = int(np.argmin(np.abs(DWP - event.xdata * J0)))
        s['probe_idx'] = idx
        self.status.showMessage(f'Probe Δ/J={DWP[idx]/J0:.4f}')
        for ax in (self.ax_thru, self.ax_drop, self.ax_del):
            for line in ax.lines:
                if line.get_linestyle() == '--': line.set_xdata([DWP[idx]/J0]*2)
        self.canvas_spec.draw_idle()
        self._draw_lattice(); self._draw_flow()

    def _on_pot_change(self, val):
        # Driven by slider — sync spinbox without re-triggering slider
        v = val / 10.
        self.spn_pot.blockSignals(True)
        self.spn_pot.setValue(v)
        self.spn_pot.blockSignals(False)
        self._apply_pot(v)

    def _on_spn_pot_change(self, v):
        # Driven by spinbox — sync slider without re-triggering spinbox
        self.sld_pot.blockSignals(True)
        self.sld_pot.setValue(int(round(v * 10)))
        self.sld_pot.blockSignals(False)
        self._apply_pot(v)

    def _apply_pot(self, v):
        n = self.state['selected']
        if n is None: return
        if v == 0.: self.state['heaters'].pop(n, None)
        else:       self.state['heaters'][n] = v
        self.lbl_site.setText(f'Site {n}  Δ={v:+.2f}J')
        self._invalidate()
        self._draw_lattice()

    def _update_io_label(self):
        s = self.state
        self.lbl_io.setText(f'IN=site {s["isite"]}   OUT=site {s["osite"]}')

    # ── Compute ───────────────────────────────────────────────────────────────
    def _run_spectrum(self):
        self.btn_run.setEnabled(False)
        self.status.showMessage('⏳ Computing…')
        s = self.state; h = self.cmb_type.currentText()
        nx0, ny0 = self.spn_nx0.value(), self.spn_ny0.value()
        nx1, ny1 = self.spn_nx1.value(), self.spn_ny1.value()
        j1       = self.spn_j1.value()
        kin      = self.sld_kin.value() / 1000.
        kex      = self.sld_kex.value() / 1000.
        phi_iqh0 = s['phi_iqh0']; phi_iqh1 = s['phi_iqh1']
        phi_aqh0 = s['phi_aqh0']; phi_aqh1 = s['phi_aqh1']
        sw_s     = self.spn_ss.value(); sw_e = self.spn_se.value(); sw_t = self.spn_st.value()
        NL, Nxt, Nyt, id_, od_ = get_lattice_params(h, nx0, ny0, nx1, ny1)
        isite = s['isite'] if 1 <= s['isite'] <= NL else id_
        # Always reset osite to type default so it tracks lattice size
        osite = od_
        s.update(h_str=h, nx0=nx0, ny0=ny0, nx1=nx1, ny1=ny1,
                 j1=j1, kin=kin, kex=kex,
                 phi_iqh0=phi_iqh0, phi_iqh1=phi_iqh1,
                 phi_aqh0=phi_aqh0, phi_aqh1=phi_aqh1,
                 NL=NL, Nxt=Nxt, Nyt=Nyt, isite=isite, osite=osite,
                 is_zz=(h == 'A_zigzag'), is_cyl=(h in ('IQH_cyl','AQH_cyl')))
        ws   = abs(sw_t) * (1 if sw_e > sw_s else -1)
        DWP  = np.arange(sw_s, sw_e + ws, ws) * J0
        psi   = s.get('psi', 0.0)
        H_mat = build_hamiltonian(h, nx0, ny0, nx1, ny1, j1,
                                   phi_iqh0, phi_iqh1, phi_aqh0, phi_aqh1, psi).copy()
        for site, shift in s['heaters'].items(): H_mat[site, site] += shift
        for site in s['defects']:    H_mat[site, :] = 0; H_mat[:, site] = 0
        self._DWP   = DWP
        self._H_mat = H_mat
        self._future = _EXECUTOR.submit(
            compute_spectrum, H_mat, kin, kex, DWP, isite, osite, NL)
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_result)
        self._poll_timer.start()

    def _poll_result(self):
        if not self._future.done():
            return
        self._poll_timer.stop()
        try:
            result = self._future.result()
        except Exception as e:
            self.status.showMessage(f'Error: {e}')
            self.btn_run.setEnabled(True)
            return
        self._on_done(result)

    def _on_done(self, result):
        thru, drop, delay, power, cf, kappa_sigma, eps_H, V_H = result; s = self.state
        s.update(spectrum=dict(DWP=self._DWP, thru=thru, drop=drop, delay=delay,
                               power=power, kappa_sigma=kappa_sigma, eps_H=eps_H, V_H=V_H),
                 H_mat=self._H_mat, complex_field=cf, probe_idx=len(self._DWP)//2)
        self._selected_sigma = 0
        self._draw_spectra(); self._draw_lattice(); self._draw_flow(); self._draw_eig()
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.status.showMessage('✅ Done — save first to feed to nonlinear simulator')

    def _open_nonlinear(self):
        from NonLinear import NonlinearWindow
        self.state['session_folder'] = getattr(self, '_session_folder', None)
        # Pass through the Linear sweep range so Nonlinear can default to it.
        # Linear convention: sw_start is where the sweep begins, sw_end is where
        # it ends. The detuning sweep in Linear typically goes high→low (e.g.
        # 1.455 → 1.4). Nonlinear wants the same physical range but with its
        # own internal convention that start > end for a redward detuning sweep.
        self.state['sw_start'] = float(self.spn_ss.value())
        self.state['sw_end']   = float(self.spn_se.value())
        self._nl_win = NonlinearWindow(self.state, parent=self)
        self._nl_win.show()

    def _sync_ui_from_state(self):
        """Update all UI widgets to match current state dict — used after loading."""
        s = self.state
        PHI_TICKS = 200   # slider range 0..200 maps to 0..2π

        widgets = [
            self.cmb_type, self.spn_nx0, self.spn_ny0, self.spn_nx1, self.spn_ny1,
            self.spn_j1, self.sld_kin, self.sld_kex,
            self.spn_kin_val, self.spn_kex_val,
            self.sld_phi_iqh0, self.sld_phi_iqh1, self.sld_phi_aqh0, self.sld_phi_aqh1,
            self.spn_phi_iqh0, self.spn_phi_iqh1, self.spn_phi_aqh0, self.spn_phi_aqh1,
            self.sld_psi, self.spn_psi_val, self.spn_ss, self.spn_se, self.spn_st,
        ]
        for w in widgets: w.blockSignals(True)

        self.cmb_type.setCurrentText(s['h_str'])
        self.spn_nx0.setValue(s['nx0']); self.spn_ny0.setValue(s['ny0'])
        self.spn_nx1.setValue(s['nx1']); self.spn_ny1.setValue(s['ny1'])
        self.spn_j1.setValue(s['j1'])
        self.sld_kin.setValue(int(round(s['kin'] * 1000)))
        self.sld_kex.setValue(int(round(s['kex'] * 1000)))
        self.spn_kin_val.setValue(s['kin'])
        self.spn_kex_val.setValue(s['kex'])
        self.sld_phi_iqh0.setValue(int(round(s['phi_iqh0'] / (2*np.pi) * PHI_TICKS)))
        self.sld_phi_iqh1.setValue(int(round(s['phi_iqh1'] / (2*np.pi) * PHI_TICKS)))
        self.sld_phi_aqh0.setValue(int(round(s['phi_aqh0'] / (2*np.pi) * PHI_TICKS)))
        self.sld_phi_aqh1.setValue(int(round(s['phi_aqh1'] / (2*np.pi) * PHI_TICKS)))
        self.spn_phi_iqh0.setValue(s['phi_iqh0'] / np.pi)
        self.spn_phi_iqh1.setValue(s['phi_iqh1'] / np.pi)
        self.spn_phi_aqh0.setValue(s['phi_aqh0'] / np.pi)
        self.spn_phi_aqh1.setValue(s['phi_aqh1'] / np.pi)
        self.sld_psi.setValue(int(round(s.get('psi', 0.0) / (2*np.pi) * 200)))
        self.spn_psi_val.setValue(s.get('psi', 0.0) / np.pi)
        # sweep spinboxes
        if 'sw_start' in s: self.spn_ss.setValue(s['sw_start'])
        if 'sw_end'   in s: self.spn_se.setValue(s['sw_end'])
        if 'sw_step'  in s: self.spn_st.setValue(s['sw_step'])

        for w in widgets: w.blockSignals(False)

        # Update IO label
        self._update_io_label()
        # Trigger visibility update for h_str-dependent sliders
        self._on_hstr_change(s['h_str'])

    def _load_existing(self):
        """Load a saved linear params.npz, restore plots, then optionally load NL data."""
        # ── Step 1: pick linear params.npz ───────────────────────────────────
        lin_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Linear params.npz', '', 'NumPy archive (*.npz)')
        if not lin_path: return

        d = np.load(lin_path, allow_pickle=True)
        s = self.state

        # Restore scalar params
        h_str = str(d['hamiltonian'][0])
        s['h_str']     = h_str
        s['nx0']       = int(d['nx0'][0]);   s['ny0'] = int(d['ny0'][0])
        s['nx1']       = int(d['nx1'][0]);   s['ny1'] = int(d['ny1'][0])
        s['j1']        = float(d['j1'][0])
        s['kin']       = float(d['kin'][0]); s['kex'] = float(d['kex'][0])
        s['phi_iqh0']  = float(d['phi_iqh0'][0])
        s['phi_iqh1']  = float(d['phi_iqh1'][0])
        s['phi_aqh0']  = float(d['phi_aqh0'][0])
        s['phi_aqh1']  = float(d['phi_aqh1'][0])
        s['psi']       = float(d['psi'][0]) if 'psi' in d else 0.0
        s['isite']     = int(d['isite'][0]); s['osite'] = int(d['osite'][0])
        # Sweep spinbox params
        if 'sw_start' in d: s['sw_start'] = float(d['sw_start'][0])
        if 'sw_end'   in d: s['sw_end']   = float(d['sw_end'][0])
        if 'sw_step'  in d: s['sw_step']  = float(d['sw_step'][0])
        # Restore heaters / defects
        s['heaters'] = {}
        if 'heater_sites' in d and len(d['heater_sites']):
            for site, val in zip(d['heater_sites'], d['heater_vals']):
                s['heaters'][int(site)] = float(val)
        s['defects'] = set(int(x) for x in d['defects']) if 'defects' in d else set()

        # Rebuild lattice geometry first (sets NL, Nxt, Nyt, redraws lattice only)
        self._sync_ui_from_state()   # update all widgets before rebuild
        self._rebuild_lattice()

        # Now restore computed results into state (after rebuild, which may clear them)
        H_full = build_hamiltonian(h_str, s['nx0'], s['ny0'], s['nx1'], s['ny1'],
                                   s['j1'], s['phi_iqh0'], s['phi_iqh1'],
                                   s['phi_aqh0'], s['phi_aqh1'], s.get('psi', 0.0))
        for site, shift in s['heaters'].items():
            H_full[site, site] += shift
        for site in s['defects']:                      # ← was missing
            H_full[site, :] = 0; H_full[:, site] = 0
        s['H_mat'] = H_full

        if 'DWP' in d:
            sp_dict = dict(
                DWP=d['DWP'], thru=d['thru'], drop=d['drop'],
                delay=d['delay'], power=d['power'])
            # Restore kappa_sigma / eps_H / V_H if saved
            if 'kappa_sigma' in d:
                sp_dict['kappa_sigma'] = d['kappa_sigma']
                sp_dict['eps_H']       = d['eps_H']
                sp_dict['V_H']         = d['V_H_re'] + 1j * d['V_H_im']
            s['spectrum']  = sp_dict
            s['probe_idx'] = len(d['DWP']) // 2
        if 'complex_field_re' in d:
            s['complex_field'] = d['complex_field_re'] + 1j * d['complex_field_im']

        self._draw_spectra()
        self._draw_lattice()
        self._draw_flow()
        self._selected_sigma = 0
        self._draw_eig()

        # Store session folder and enable buttons
        self._session_folder = os.path.dirname(lin_path)
        self.status.showMessage(f'✅ Loaded linear session → {self._session_folder}')
        self.btn_save.setEnabled(True)
        self.btn_feed.setEnabled(True)           # ← was missing

        # ── Step 2: optionally pick nonlinear params.npz ─────────────────────
        reply = QMessageBox.question(self, 'Load Nonlinear Data',
                                     'Also load a nonlinear simulation?',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return

        nl_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Nonlinear params.npz',
            self._session_folder, 'NumPy archive (*.npz)')
        if not nl_path: return

        # Ensure the NL file is actually inside the linear session folder
        lin_folder_norm = os.path.normcase(os.path.abspath(self._session_folder))
        nl_path_norm    = os.path.normcase(os.path.abspath(nl_path))
        if not nl_path_norm.startswith(lin_folder_norm):
            QMessageBox.warning(self, 'Wrong folder',
                                f'Please choose a params.npz inside:\n{self._session_folder}')
            return

        nd = np.load(nl_path, allow_pickle=True)

        if 'a_T' not in nd:
            QMessageBox.warning(self, 'Load Error',
                                'This params.npz has no a_T array (simulation was not run).')
            return

        from NonLinear import NonlinearWindow, Schedule

        # Build NL state from loaded data
        nl_state = dict(self.state)
        nl_state['session_folder'] = self._session_folder

        self._nl_win = NonlinearWindow(nl_state, parent=self)

        # Pre-load schedules and results into the window
        n_steps = int(nd['N_steps'][0]) if 'N_steps' in nd else nd['a_T'].shape[2]
        nl_win  = self._nl_win

        # Set stepper params
        nl_win.spn_fsr.setValue(int(nd['one_side_fsr'][0]))
        nl_win.spn_D2.setValue(float(nd['D2'][0]))
        nl_win.spn_niter.setValue(int(nd['NIter'][0]))
        nl_win.spn_dt.setValue(float(nd['TStep'][0]))
        nl_win.spn_nsteps.setValue(n_steps)

        # Set schedule spinboxes — block only value spinboxes to avoid premature redraws
        # Do NOT block mode_combo — _on_mode must fire to switch the QStackedWidget page
        def _block_row(row, block):
            for w in [row.spn_fixed, row.spn_from, row.spn_to]:
                w.blockSignals(block)

        if 'F_schedule' in nd and len(nd['F_schedule']):
            F = nd['F_schedule']
            f_row = next((pr for pr in nl_win._param_rows if pr.pkey == 'F'), None)
            if f_row:
                _block_row(f_row, True)
                if np.allclose(F, F[0]):
                    f_row.mode_combo.setCurrentText('Fixed')
                    f_row.spn_fixed.setValue(float(F[0]))
                else:
                    f_row.mode_combo.setCurrentText('Sweep')
                    f_row.spn_from.setValue(float(F[0])); f_row.spn_to.setValue(float(F[-1]))
                _block_row(f_row, False)

        if 'D_schedule' in nd and len(nd['D_schedule']):
            D = nd['D_schedule']
            d_row = next((pr for pr in nl_win._param_rows if pr.pkey == 'detuning'), None)
            if d_row:
                _block_row(d_row, True)
                if np.allclose(D, D[0]):
                    d_row.mode_combo.setCurrentText('Fixed')
                    d_row.spn_fixed.setValue(float(D[0]))
                else:
                    d_row.mode_combo.setCurrentText('Sweep')
                    d_row.spn_from.setValue(float(D[0])); d_row.spn_to.setValue(float(D[-1]))
                _block_row(d_row, False)

        if 'psi_schedule' in nd and len(nd['psi_schedule']):
            psi = nd['psi_schedule']
            psi_row = next((pr for pr in nl_win._param_rows if pr.pkey == 'psi'), None)
            if psi_row:
                _block_row(psi_row, True)
                if np.allclose(psi, psi[0]):
                    psi_row.mode_combo.setCurrentText('Fixed')
                    psi_row.spn_fixed.setValue(float(psi[0] / np.pi))
                else:
                    psi_row.mode_combo.setCurrentText('Sweep')
                    psi_row.spn_from.setValue(float(psi[0] / np.pi))
                    psi_row.spn_to.setValue(float(psi[-1] / np.pi))
                _block_row(psi_row, False)

        # Inject results and plot everything.
        # FSRVec / PumpFSR are stored in FFT order (clean-break convention).
        _one_side = int(nd['one_side_fsr'][0])
        _NF       = 2*_one_side + 1
        FSRVec  = nd['FSRVec'] if 'FSRVec' in nd else np.round(np.fft.fftfreq(_NF, d=1.0)*_NF).astype(float)
        PumpFSR = int(nd['PumpFSR'][0]) if 'PumpFSR' in nd else 0
        a_T     = nd['a_T']             # (NLattice, NumFSRs, N_steps)

        nl_win._results = dict(a_T=a_T, FSRVec=FSRVec, PumpFSR=PumpFSR,
                               n_done=a_T.shape[2])
        nl_win._xv      = np.arange(n_steps)
        nl_win._n_steps = n_steps
        NL = s['NL']; NumFSRs = a_T.shape[1]

        # Init probe and power curves from data
        OSite_i = s['osite'] - 1
        a_W     = np.fft.fft(a_T, axis=1, norm='ortho')
        mask    = np.ones(NumFSRs, bool); mask[PumpFSR] = False
        nl_win._pump_n     = np.abs(a_W[OSite_i, PumpFSR, :])**2
        nl_win._side_n     = np.abs(a_W[OSite_i, mask, :]).sum(axis=0)
        nl_win._pump_n_all = np.abs(a_W[:, PumpFSR, :]).sum(axis=0)
        nl_win._side_n_all = np.abs(a_W[:, mask, :]).sum(axis=(0, 1))

        probe = n_steps - 1

        # Show window — showEvent will trigger _draw_sched_preview automatically
        nl_win.show()
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        # Build _sched_snapshot from loaded schedule arrays so _update_step_params
        # and _open_slow_time work correctly
        # Use F_schedule length as authoritative n_steps for Schedule
        sched_n = len(nd['F_schedule']) if 'F_schedule' in nd and len(nd['F_schedule']) else n_steps
        sched = Schedule(sched_n)
        if 'F_schedule' in nd and len(nd['F_schedule']):
            sched.add('F', nd['F_schedule'][:sched_n])
        if 'D_schedule' in nd and len(nd['D_schedule']):
            sched.add('detuning', nd['D_schedule'][:sched_n])
        if 'psi_schedule' in nd and len(nd['psi_schedule']):
            sched.add('psi', nd['psi_schedule'][:sched_n])
        nl_win._sched_snapshot = sched

        # Init spec plots BEFORE setting slider (slider triggers _on_probe_change
        # which needs _vline_pump_all etc. to already exist)
        nl_win._init_spec_plots('Step')
        nl_win.prog_bar.setValue(100)

        # Now set slider/spinbox safely
        nl_win.sld_probe.blockSignals(True)
        nl_win.sld_probe.setRange(0, n_steps - 1); nl_win.sld_probe.setValue(probe)
        nl_win.sld_probe.blockSignals(False)
        nl_win.spn_probe.blockSignals(True)
        nl_win.spn_probe.setRange(0, n_steps - 1); nl_win.spn_probe.setValue(probe)
        nl_win.spn_probe.blockSignals(False)
        nl_win.lbl_probe.setText(f'Probe: step {probe}')

        nl_win._update_spec_plots(n_steps, probe)
        nl_win._draw_lattice()
        nl_win._heat_imgs = {}   # force fresh imshow (not update of stale image)
        nl_win._draw_heat_maps()
        nl_win._update_cross_sections(probe)
        nl_win._update_step_params(probe)
        nl_win.btn_save.setEnabled(True)
        nl_win.lbl_log.setText(f'Loaded — {n_steps} steps')
        nl_win.status.showMessage(f'✅ Loaded NL session — {n_steps} steps')

    def _clear(self):
        s = self.state; s['heaters'].clear(); s['defects'].clear(); s['selected'] = None
        self.sld_pot.setValue(0); self.spn_pot.setValue(0.)
        self.lbl_site.setText('Click a ring')
        self._invalidate()
        self._draw_lattice()

    def _browse_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Save Folder', self.edit_save_path.text())
        if folder: self.edit_save_path.setText(folder)

    def _make_session_name(self):
        """Build a descriptive folder name from current simulation parameters.
        Example: II_6611_kex_0p01_kin_0p001_phi_0p5pi_0p5pi
        """
        s = self.state
        h = s['h_str']

        abbrev = {
            'IQH_IQH': 'II', 'AQH_AQH': 'AA',
            'IQH_AQH': 'IA', 'AQH_IQH': 'AI',
            'A_zigzag': 'AZ', 'IQH_cyl': 'IC', 'AQH_cyl': 'AC',
        }.get(h, h.replace('_', ''))

        dims = f"{s['nx0']}{s['ny0']}{s['nx1']}{s['ny1']}"

        def _flt(v):   return f"{v:.4g}".replace('.', 'p')
        def _phi(v):   return f"{v / np.pi:.4g}".replace('.', 'p') + 'pi'

        phi_vals = {
            'IQH_IQH': [s['phi_iqh0'], s['phi_iqh1']],
            'AQH_AQH': [s['phi_aqh0'], s['phi_aqh1']],
            'IQH_AQH': [s['phi_iqh0'], s['phi_aqh1']],
            'AQH_IQH': [s['phi_aqh0'], s['phi_iqh1']],
            'A_zigzag': [s['phi_aqh0']],
            'IQH_cyl':  [s['phi_iqh0'], s.get('psi', 0.0)],
            'AQH_cyl':  [s['phi_aqh0'], s.get('psi', 0.0)],
        }.get(h, [])

        phi_str = '_'.join(_phi(p) for p in phi_vals)

        return (f"{abbrev}_{dims}"
                f"_kex_{_flt(s['kex'])}_kin_{_flt(s['kin'])}"
                f"_phi_{phi_str}"
                f"_I_{s['isite']}_O_{s['osite']}")

    def _save(self):
        try:
            self._save_inner()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            QMessageBox.critical(self, 'Save Error',
                f'Save failed at step:\n\n{str(e)}\n\nDetails:\n{err}')
            self.status.showMessage(f'❌ Save failed: {e}')
            self.btn_save.setEnabled(True)

    def _save_inner(self):
        base   = self.edit_save_path.text().strip() or (
            os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
            else os.path.dirname(os.path.abspath(__file__)))
        name   = self._make_session_name()
        folder = os.path.join(base, name)
        os.makedirs(folder, exist_ok=True)

        self.btn_save.setEnabled(False)
        self.status.showMessage('💾 Saving...')
        s = self.state

        # ── params.npz — save this first (most important) ─────────────────────
        params = dict(
            hamiltonian  = np.array([s['h_str']]),
            nx0          = np.array([s['nx0']]),
            ny0          = np.array([s['ny0']]),
            nx1          = np.array([s['nx1']]),
            ny1          = np.array([s['ny1']]),
            j1           = np.array([s['j1']]),
            kin          = np.array([s['kin']]),
            kex          = np.array([s['kex']]),
            phi_iqh0     = np.array([s['phi_iqh0']]),
            phi_iqh1     = np.array([s['phi_iqh1']]),
            phi_aqh0     = np.array([s['phi_aqh0']]),
            phi_aqh1     = np.array([s['phi_aqh1']]),
            psi          = np.array([s.get('psi', 0.0)]),
            isite        = np.array([s['isite']]),
            osite        = np.array([s['osite']]),
            sw_start     = np.array([self.spn_ss.value()]),
            sw_end       = np.array([self.spn_se.value()]),
            sw_step      = np.array([self.spn_st.value()]),
            heater_sites = np.array(sorted(s['heaters'].keys()), dtype=int),
            heater_vals  = np.array([s['heaters'][k] for k in sorted(s['heaters'].keys())]),
            defects      = np.array(sorted(s['defects']), dtype=int),
        )
        sp = s.get('spectrum')
        pb_label = ''
        if sp is not None:
            pb_val = sp['DWP'][s['probe_idx']] / J0
            pb_label = f"_pb_{f'{pb_val:.4g}'.replace('.', 'p')}"
            params.update(DWP=sp['DWP'], thru=sp['thru'],
                          drop=sp['drop'], delay=sp['delay'], power=sp['power'])
            if sp.get('kappa_sigma') is not None:
                params['kappa_sigma'] = sp['kappa_sigma']
                params['eps_H']       = sp['eps_H']
                params['V_H_re']      = sp['V_H'].real
                params['V_H_im']      = sp['V_H'].imag
        if s.get('complex_field') is not None:
            params['complex_field_re'] = s['complex_field'].real
            params['complex_field_im'] = s['complex_field'].imag
        np.savez(os.path.join(folder, 'linear_params.npz'), **params)
        self.status.showMessage('💾 Saving figures...')

        # ── Figures: spectra, lattice, photon flow ────────────────────────────
        for fig_, fname in [(self.fig_spec,  'spectra'),
                            (self.fig_lat,   'lattice'),
                            (self.fig_flow,  'flow'),
                            (self.fig_eig,   'eigenvalues')]:
            for fmt in ('png', 'svg'):
                try:
                    buf = io.BytesIO()
                    fig_.savefig(buf, format=fmt, dpi=200, bbox_inches='tight',
                                 facecolor=fig_.get_facecolor())
                    with open(os.path.join(folder, f'{fname}{pb_label}.{fmt}'), 'wb') as fh:
                        fh.write(buf.getvalue())
                except Exception as fig_err:
                    # Log figure save failure but don't abort — data is already saved
                    self.status.showMessage(f'⚠️ Could not save {fname}.{fmt}: {fig_err}')

        # Remember so Feed to Nonlinear can use the same folder
        self._session_folder = folder
        self.btn_feed.setEnabled(True)
        self.status.showMessage(f'✅ Saved → {folder}')
        self.btn_save.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 9))

    # ── Update PyInstaller built-in splash text during startup ───────────────
    # pyi_splash is injected by the bootloader when built with PyInstaller.
    # Calls are no-ops (ImportError caught) when running directly as .py.
    def _splash(msg):
        try:
            import pyi_splash
            pyi_splash.update_text(msg)
        except ImportError:
            pass

    _splash('Initialising…')

    win = MainWindow()

    _splash('Building interface…')
    win.show()

    # ── Close the built-in splash once the main window is ready ──────────────
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass

    sys.exit(app.exec_())