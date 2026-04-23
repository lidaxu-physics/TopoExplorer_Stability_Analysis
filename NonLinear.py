"""
NonLinear.py
============
Nonlinear frequency-comb simulation window, opened from the linear
Topological Photonic Lattice Explorer (Linear.py) via the
"Feed to Nonlinear Simulator" button.

Usage (inside Linear.py):
    from NonLinear import NonlinearWindow
    self._nl_win = NonlinearWindow(self.state, parent=self)
    self._nl_win.show()
"""

import os, io, sys, zipfile, datetime, queue, threading
import numpy as np
import scipy.linalg

# Import lattice Hamiltonian builder from Linear.py (same directory)
try:
    import importlib.util as _ilu, sys as _sys
    _base = (sys._MEIPASS if getattr(sys, 'frozen', False)
             else os.path.dirname(os.path.abspath(__file__)))
    _spec = _ilu.spec_from_file_location(
        'Linear', os.path.join(_base, 'Linear.py'))
    _lin = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_lin)
    build_hamiltonian = _lin.build_hamiltonian
except Exception:
    build_hamiltonian = None   # fallback: ψ sweep without precompute

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QGroupBox, QSplitter, QStatusBar, QFileDialog,
    QProgressBar, QScrollArea, QSizePolicy, QFrame,
    QDialog, QStackedWidget, QPlainTextEdit, QMessageBox, QTabWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui  import QFont

import matplotlib
if matplotlib.get_backend() != 'Qt5Agg':
    matplotlib.use('Qt5Agg')      # must be set before any pyplot import; guard against double-set
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import LineCollection

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    jax.config.update("jax_enable_x64", True)
    # Runtime test: verify XLA can actually JIT-compile and execute — not just import.
    # In a PyInstaller exe, 'import jax' succeeds even when XLA data files (MLIR
    # dialects, kernel libraries) are missing from the bundle.  The crash happens
    # silently the first time jit() tries to lower a function.  Running a trivial
    # jit here at startup catches that failure as a Python exception so we can fall
    # back to NumPy instead of segfaulting mid-simulation when Run is clicked.
    _jax_probe = jit(lambda x: x + 1.0)
    _jax_probe(jnp.array(1.0, dtype=jnp.float64))
    del _jax_probe
    JAX_AVAILABLE = True
except Exception:
    # Covers ImportError, any XLA initialisation error, and jit compile failures.
    JAX_AVAILABLE = False
    jnp = np          # fallback: jnp aliases numpy so non-jit code still runs
#if getattr(sys, 'frozen', False):
#    JAX_AVAILABLE = False
# =====================================================================
# LATTICE GEOMETRY  (mirrored from Linear.py for visualization)
# =====================================================================
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

def site_xy(n, h_str, nx0, ny0, nx1, ny1):
    if h_str == 'A_zigzag':
        loc = NumberToLocation_AQH_zigzag(n, nx0, ny0)
        return float(2 * loc[0] - 1 + loc[1] % 2), float(loc[1])
    if h_str in ('IQH_cyl', 'AQH_cyl'):
        mx, my = NumberToLocation(n, nx0, ny0)
        R0    = nx0 / (2 * np.pi)
        theta = (mx - 1) / nx0 * 2 * np.pi
        r     = R0 + (ny0 - my) * 1.2
        return float(r * np.cos(theta)), float(r * np.sin(theta))
    sl1, sl0 = superlattice(nx0, ny0, n)
    x1, y1   = NumberToLocation(sl1, nx1, ny1)
    x0, y0   = NumberToLocation(sl0, nx0, ny0)
    return float((x1 - 1) * nx0 + x0 - 1), float((y1 - 1) * ny0 + y0 - 1)


def compute_flow(field_1d, H_mat, NL, isite, osite, h_str, nx0, ny0, nx1, ny1,
                 normalize=True):
    """Photon current arrows for all lattice types.
    Current from site n to nb: J = -Im(ψ_n* H_{n,nb} ψ_{nb})
    Mirrored from Linear.py with the correct sign convention.

    normalize=True (default, backward compatible): rescale so the largest
    arrow is unit length. Use this for single-frame plots.

    normalize=False: return the raw (unnormalized) U,V so the caller can
    apply a global scale across many frames. Used by SlowTimeWindow to
    make arrow lengths comparable across frames in animations.
    """
    is_zz  = (h_str == 'A_zigzag')
    is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
    Nxt    = nx0 if (is_zz or is_cyl) else nx0 * nx1
    Nyt    = ny0 if (is_zz or is_cyl) else ny0 * ny1
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
            arr = np.zeros(2)
            for v, u in zip(vecs, norms):
                xb, yb = xn + v[0], yn + v[1]
                if not inlat(xb, yb): continue
                try:
                    nb = c2s(xb, yb)
                    f  = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')

    elif is_cyl:
        sc = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL + 1)}
        for n in range(1, NL + 1):
            xn, yn = sc[n]
            arr = np.zeros(2)
            for nb in range(1, NL + 1):
                if nb == n or H_mat[n, nb] == 0: continue
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
            arr = np.zeros(2)
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
    if normalize:
        mx = np.sqrt(U ** 2 + V ** 2).max()
        if mx > 0: U /= mx; V /= mx
    U = -U; V = -V
    return (np.array(Xr), np.array(Yr), U, V, cols)


# =====================================================================
# PHYSICS ENGINE
# =====================================================================

def build_disp_loss(NLattice, one_side_fsr, D2, kin, kex, ISite, OSite):
    """
    FFT-ORDER CONVENTION (Option A):
      FSRVec[0]  = 0  (DC / pump)
      FSRVec[1]  = +1, FSRVec[2] = +2, ..., FSRVec[one_side] = +one_side
      FSRVec[one_side+1] = -one_side, ..., FSRVec[NumFSRs-1] = -1
    i.e. the same ordering as np.fft.fftfreq(NumFSRs) * NumFSRs.
    The pump mode (μ=0) is always at index 0, so PumpFSR = 0.
    """
    NumFSRs = 2*one_side_fsr + 1
    # FFT-ordered integer μ values: [0, 1, ..., +N, -N, ..., -1]
    # fftfreq gives float; round to avoid tiny precision artefacts like 2.9999999…
    FSRVec  = np.round(np.fft.fftfreq(NumFSRs, d=1.0) * NumFSRs).astype(float)
    DispMat = np.tile(FSRVec**2*(D2/2.), (NLattice,1)).astype(np.complex128)
    LM      = np.full((NLattice, NumFSRs), kin, dtype=np.complex128)
    LM[ISite-1, :] += kex
    if OSite != ISite: LM[OSite-1, :] += kex
    PumpFSR = 0
    return DispMat, LM, FSRVec, PumpFSR, NumFSRs


def build_propagators(H, DispMat, LM, detuning, dt, F, PumpFSR, ISite):
    NLattice = H.shape[0];  NumFSRs = DispMat.shape[1]
    props = np.empty((NumFSRs, NLattice, NLattice), dtype=np.complex128)
    fc    = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    eye   = np.eye(NLattice, dtype=np.complex128);  cache = {}
    for mu in range(NumFSRs):
        M = np.diag((1j*detuning - 1j*DispMat[:,mu] - LM[:,mu]).astype(np.complex128)) - 1j*H
        P = scipy.linalg.expm(M*dt);  props[mu] = P
        if mu == PumpFSR:
            dP = P - eye
            try:    MidP = np.linalg.solve(M, dP)
            except Exception: MidP = np.linalg.lstsq(M, dP, rcond=None)[0]
            cache['MidP_pump'] = MidP
            fv = np.zeros(NLattice, dtype=np.complex128);  fv[ISite-1] = F
            fc[mu] = MidP @ fv
    return props, fc, cache


# Module-level cache for build_propagators_fast eigendecomposition results.
# Keyed by (H_bytes, shape) so identity is based on content, not object id().
_BPF_EIG_CACHE: dict = {}

def build_propagators_fast(H, DispMat, LM, detuning, dt, F, PumpFSR, ISite):
    """Fast propagator build using eigendecomposition of H_aug.

    Replaces 257 expm calls with 1 eig + 257 vector-exp + 257 matrix multiplies.
    ~20-30x faster than build_propagators when H changes (ψ / heater sweeps).

    H_aug = H - i·kex·diag(IO_mask)  absorbs the site-dependent loss.
    M_μ   = c_μ·I - i·H_aug  where c_μ is a scalar (uniform per mode).
    expm(M_μ·dt) = V · diag(exp((c_μ - i·λ)·dt)) · V⁻¹
    """
    NLattice = H.shape[0]; NumFSRs = DispMat.shape[1]

    # Extra loss beyond uniform kin (site-dependent part)
    kex_per_site = LM[:, 0] - LM[:, 0].min()   # shape (NL,), real
    H_aug = H - 1j * np.diag(kex_per_site)       # absorb into H_aug
    kin   = LM[:, 0].min().real                   # uniform loss floor

    # Eigendecompose H_aug (non-Hermitian in general).
    # Cache keyed by tobytes() + shape — stable content-based identity,
    # avoids the id()-reuse footgun of the old mutable default arg approach.
    H_key = (H_aug.tobytes(), H_aug.shape)
    if H_key not in _BPF_EIG_CACHE:
        lam, V = scipy.linalg.eig(H_aug)
        Vd = np.linalg.inv(V)
        if len(_BPF_EIG_CACHE) > 8:   # bound memory: keep at most 8 entries
            _BPF_EIG_CACHE.pop(next(iter(_BPF_EIG_CACHE)))
        _BPF_EIG_CACHE[H_key] = (lam, V, Vd)
    lam, V, Vd = _BPF_EIG_CACHE[H_key]

    # Uniform diagonal scalar per mode: c_μ = i·Δ - i·D2·μ²/2 - kin
    # DispMat[m, mu] = D2/2 · FSRVec[mu]^2 (same for all m)
    disp_per_mode = DispMat[0, :]   # (NumFSRs,) — use row 0 (same for all sites)
    c_mus = 1j*detuning - 1j*disp_per_mode - kin   # (NumFSRs,) complex

    # Build all props at once: props[mu] = V @ diag(exp((c_mu - i*lam)*dt)) @ Vd
    # exp_all: (NumFSRs, NL)
    exp_all = np.exp((c_mus[:, np.newaxis] - 1j*lam[np.newaxis, :]) * dt)  # (NumFSRs, NL)
    # props[mu] = (V * exp_all[mu,:]) @ Vd
    props = np.einsum('kl,ml,lj->mkj', V, exp_all, Vd)   # (NumFSRs, NL, NL)

    # Force real symmetry (tiny imaginary artifacts from eig)
    # props = props.real + 1j*props.imag  — keep as-is, already complex

    # Pump forcing fc: same as build_propagators
    fc    = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    eye   = np.eye(NLattice, dtype=np.complex128); cache = {}
    # Recompute M at pump mode only (for MidP)
    mu_p   = PumpFSR
    M_pump = c_mus[mu_p]*eye - 1j*H_aug
    dP     = props[mu_p] - eye
    try:    MidP = np.linalg.solve(M_pump, dP)
    except Exception: MidP = np.linalg.lstsq(M_pump, dP, rcond=None)[0]
    cache['MidP_pump'] = MidP
    fv = np.zeros(NLattice, dtype=np.complex128); fv[ISite-1] = F
    fc[mu_p] = MidP @ fv

    return props, fc, cache


def update_fc_fast(cache, F, PumpFSR, ISite, NumFSRs, NLattice):
    """Cheapest update: only the pump forcing vector changed (F changed, H and M unchanged)."""
    fc   = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    MidP = cache.get('MidP_pump')
    if MidP is not None:
        fv = np.zeros(NLattice, dtype=np.complex128);  fv[ISite-1] = F
        fc[PumpFSR] = MidP @ fv
    return fc


_stepper_cache: dict = {}

def get_stepper(NIter: int):
    """Return a compiled JAX stepper if JAX is available, else a plain NumPy loop."""
    if NIter not in _stepper_cache:
        if JAX_AVAILABLE:
            @partial(jit)
            def _run(a_T, props, fc, dt):
                def step(a_T, _):
                    a_T = jnp.exp(1j*(dt*0.5)*jnp.abs(a_T)**2) * a_T
                    a_W = jnp.fft.fft(a_T, axis=1, norm='ortho')
                    a_W = jnp.einsum('mij,jm->im', props, a_W) + fc.T
                    a_T = jnp.fft.ifft(a_W, axis=1, norm='ortho')
                    a_T = jnp.exp(1j*(dt*0.5)*jnp.abs(a_T)**2) * a_T
                    return a_T, None
                a_final, _ = jax.lax.scan(step, a_T, None, length=NIter)
                return a_final
        else:
            def _run(a_T, props, fc, dt):
                """Pure-NumPy fallback stepper (no JIT compilation)."""
                a_T = np.array(a_T, dtype=complex)
                for _ in range(NIter):
                    a_T = np.exp(1j*(dt*0.5)*np.abs(a_T)**2) * a_T
                    a_W = np.fft.fft(a_T, axis=1, norm='ortho')
                    a_W = np.einsum('mij,jm->im', np.array(props), a_W) + np.array(fc).T
                    a_T = np.fft.ifft(a_W, axis=1, norm='ortho')
                    a_T = np.exp(1j*(dt*0.5)*np.abs(a_T)**2) * a_T
                return a_T
        _stepper_cache[NIter] = _run
    return _stepper_cache[NIter]


def make_initial_state(NLattice, NumFSRs, seed=42):
    rng = np.random.default_rng(seed)
    a_W = 1e-8*rng.random((NLattice,NumFSRs))*np.exp(1j*rng.random((NLattice,NumFSRs))*2*np.pi)
    state = np.fft.ifft(a_W, axis=1, norm='ortho')
    return jnp.array(state) if JAX_AVAILABLE else state


_H_KEYS = {'psi'}


class Schedule:
    def __init__(self, n_steps):
        self.n_steps  = int(n_steps)
        self.channels = {}

    def add(self, name, values):
        arr = np.asarray(values, dtype=float)
        if arr.shape != (self.n_steps,):
            raise ValueError(f"Channel '{name}': shape mismatch")
        self.channels[name] = arr;  return self

    def ramp(self, name, start, end):
        return self.add(name, np.linspace(start, end, self.n_steps))

    def fixed(self, name, value):
        return self.add(name, np.full(self.n_steps, float(value)))

    def at(self, gg):
        return {k: float(v[gg]) for k, v in self.channels.items()}

    def _changed(self, gg, key):
        if key not in self.channels: return False
        return gg == 0 or self.channels[key][gg] != self.channels[key][gg-1]

    def needs_H_rebuild(self, gg):
        hk = _H_KEYS | {k for k in self.channels if k.startswith('heater_')}
        return any(self._changed(gg, k) for k in hk)

    def needs_M_rebuild(self, gg):
        return self.needs_H_rebuild(gg) or self._changed(gg, 'detuning')

    def needs_fc_only(self, gg):
        return not self.needs_M_rebuild(gg) and self._changed(gg, 'F')


class SweepRunner:
    def __init__(self, fp):
        self.fp = dict(fp)

    def run(self, schedule, callback=None):
        import time
        fp       = self.fp
        NIter    = int(fp['NIter']);    dt       = float(fp['TStep'])
        ISite    = int(fp['ISite']);    OSite    = int(fp['OSite'])
        NLattice = int(fp['NLattice'])
        H_base   = fp['H_mat'].copy()

        DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
            NLattice, fp['one_side_fsr'], fp['D2'], fp['kin'], fp['kex'], ISite, OSite)

        stepper = get_stepper(NIter)
        a_T     = make_initial_state(NLattice, NumFSRs)
        H       = H_base.copy()
        props   = fc_arr = cache = props_j = fc_j = None
        a_T_all = np.zeros((NLattice, NumFSRs, schedule.n_steps), dtype=np.complex128)
        t0 = time.time()
        last_gg = -1

        precomp_props = precomp_fc = None   # unused, kept for future reference

        try:
            for gg in range(schedule.n_steps):
                sp       = schedule.at(gg)
                detuning = sp.get('detuning', fp.get('detuning', 1.5))
                F        = sp.get('F',        fp.get('F',        0.3))

                if schedule.needs_H_rebuild(gg):
                    # Rebuild H: apply new ψ and/or heaters
                    psi_val = sp.get('psi', 0.0)
                    if 'h_str' in fp and build_hamiltonian is not None:
                        H = build_hamiltonian(
                            fp['h_str'], fp['nx0'], fp['ny0'], fp['nx1'], fp['ny1'],
                            fp['j1'], fp['phi_iqh0'], fp['phi_iqh1'],
                            fp['phi_aqh0'], fp['phi_aqh1'], psi_val)[1:, 1:]
                    else:
                        H = H_base.copy()
                    for k, v in sp.items():
                        if k.startswith('heater_'):
                            n = int(k.split('_')[1]); H[n-1, n-1] += v
                    # Fast eigendecomposition-based propagator (~20x faster than expm)
                    props, fc_arr, cache = build_propagators_fast(
                        H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                    props_j = jnp.array(props); fc_j = jnp.array(fc_arr)

                elif props is None or schedule.needs_M_rebuild(gg):
                    # Detuning changed only — standard build
                    props, fc_arr, cache = build_propagators(
                        H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                    props_j = jnp.array(props); fc_j = jnp.array(fc_arr)

                elif schedule.needs_fc_only(gg):
                    # Only F changed — cheapest update
                    fc_arr = update_fc_fast(cache, F, PumpFSR, ISite, NumFSRs, NLattice)
                    fc_j   = jnp.array(fc_arr)

                a_T    = stepper(a_T, props_j, fc_j, dt)
                a_T_np = np.array(a_T); a_T_all[:, :, gg] = a_T_np
                last_gg = gg

                if callback is not None:
                    callback(gg, a_T_np, sp, time.time()-t0)

        except StopIteration:
            pass

        n_done = last_gg + 1
        return dict(a_T=a_T_all[:, :, :n_done], FSRVec=FSRVec,
                    PumpFSR=PumpFSR, schedule=schedule, n_done=n_done)


# =====================================================================
# SECTION 7 – QT APPLICATION
# =====================================================================

DARK_BG   = '#08090d';  PANEL_BG = '#0a0c14';  GRID_COL  = '#1a1f2e'
SPINE_COL = '#1e2230';  TEXT_DIM = '#4a5270';  TEXT_COL  = '#c8d0e7'
ACCENT    = '#00e5ff';  RING_R   = 0.30

STYLE = """
QMainWindow,QWidget{background:#08090d;color:#c8d0e7;font-size:14px;}
QGroupBox{border:1px solid #1e2230;border-radius:6px;margin-top:10px;
          padding:7px;font-size:13px;font-weight:bold;color:#3a4a70;}
QGroupBox::title{subcontrol-origin:margin;left:8px;}
QPushButton{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
            padding:5px 11px;color:#c8d0e7;font-size:13px;}
QPushButton:hover{background:#1e2a40;border-color:#00e5ff;}
QPushButton:pressed,QPushButton:checked{background:#00e5ff;color:#08090d;}
QPushButton:disabled{color:#2a3050;}
QComboBox,QDoubleSpinBox,QSpinBox{background:#171c2e;border:1px solid #1e2230;
    border-radius:4px;padding:4px 6px;color:#c8d0e7;font-size:13px;}
QSlider::groove:horizontal{height:4px;background:#1e2230;border-radius:2px;}
QSlider::handle:horizontal{background:#00e5ff;width:14px;height:14px;
    margin:-5px 0;border-radius:7px;}
QProgressBar{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
             color:#4caf50;font-size:12px;font-weight:bold;text-align:center;}
QProgressBar::chunk{background:#00e5ff;border-radius:3px;}
QLabel{color:#c8d0e7;font-size:13px;}
QStatusBar{color:#4a5270;font-size:12px;}
QScrollArea{border:none;}
QPlainTextEdit{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
               color:#c8d0e7;font-size:12px;font-family:Courier New;}
"""

CYL_TYPES = {'IQH_cyl', 'AQH_cyl'}


def _phi_str(v_rad):
    ticks = round(v_rad / np.pi * 100)
    m = {0:'0', 25:'pi/4', 33:'pi/3', 50:'pi/2', 67:'2pi/3',
         75:'3pi/4', 100:'pi', 150:'3pi/2', 200:'2pi'}
    return m.get(ticks, f'{v_rad:.4f} rad')


# =====================================================================
# DESIGN ARRAY DIALOG
# =====================================================================
class DesignArrayDialog(QDialog):
    def __init__(self, n_steps, label, current_array=None, parent=None):
        super().__init__(parent)
        self.n_steps      = n_steps
        self.result_array = current_array.copy() if current_array is not None else None
        self.setWindowTitle(f'Design schedule:  {label}')
        self.setMinimumSize(520, 440)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self);  lay.setSpacing(8)
        info = QLabel(f'Enter exactly {n_steps} values  (comma-separated or one per line):')
        info.setStyleSheet('color:#c8d0e7;font-size:13px;')
        lay.addWidget(info)

        self.text_edit = QPlainTextEdit()
        if current_array is not None:
            self.text_edit.setPlainText(', '.join(f'{v:.4f}' for v in current_array))
        else:
            self.text_edit.setPlaceholderText(
                'Example:  1.8, 1.75, 1.70, ...\nor one value per line')
        lay.addWidget(self.text_edit)

        self.fig_p    = Figure(figsize=(5, 2), facecolor=DARK_BG, tight_layout=False)
        self.canvas_p = FigureCanvas(self.fig_p)
        self.canvas_p.setFixedHeight(160)
        self.ax_p = self.fig_p.add_subplot(111)
        self.ax_p.set_facecolor(PANEL_BG)
        for sp in self.ax_p.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_p.tick_params(colors=TEXT_COL, labelsize=9)
        self.ax_p.set_xlabel('Step', color=TEXT_DIM, fontsize=10)
        self.ax_p.grid(True, color=GRID_COL, lw=0.4)
        self.fig_p.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.22)
        lay.addWidget(self.canvas_p)

        self.lbl_status = QLabel('');  self.lbl_status.setStyleSheet('font-size:12px;')
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_prev = QPushButton('Preview');  btn_prev.clicked.connect(self._preview)
        btn_ok   = QPushButton('OK');       btn_ok.clicked.connect(self._accept)
        btn_can  = QPushButton('Cancel');   btn_can.clicked.connect(self.reject)
        btn_row.addWidget(btn_prev);  btn_row.addStretch()
        btn_row.addWidget(btn_ok);    btn_row.addWidget(btn_can)
        lay.addLayout(btn_row)

        if current_array is not None:
            self._preview()

    def _parse(self):
        text = self.text_edit.toPlainText().strip()
        text = text.replace('\n', ',').replace(';', ',')
        parts = [p.strip() for p in text.split(',') if p.strip()]
        try:    return [float(p) for p in parts]
        except Exception: return None

    def _preview(self):
        vals = self._parse()
        if vals is None:
            self.lbl_status.setStyleSheet('color:#ff4a6e;font-size:12px;')
            self.lbl_status.setText('Error: could not parse — check for non-numeric entries')
            return
        color = '#a8ff78' if len(vals)==self.n_steps else '#ffcc00'
        self.lbl_status.setStyleSheet(f'color:{color};font-size:12px;')
        self.lbl_status.setText(f'{len(vals)} values parsed' +
            ('' if len(vals)==self.n_steps
             else f'  (need {self.n_steps} — will interpolate on OK)'))
        self.ax_p.cla();  self.ax_p.set_facecolor(PANEL_BG)
        self.ax_p.plot(np.arange(len(vals)), vals, color=ACCENT, lw=1.4)
        self.ax_p.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax_p.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_p.tick_params(colors=TEXT_COL, labelsize=9)
        self.canvas_p.draw_idle()

    def _accept(self):
        vals = self._parse()
        if vals is None:
            self.lbl_status.setStyleSheet('color:#ff4a6e;font-size:12px;')
            self.lbl_status.setText('Error: could not parse values');  return
        arr = np.array(vals, dtype=float)
        if len(arr) != self.n_steps:
            x_old = np.linspace(0, 1, len(arr));  x_new = np.linspace(0, 1, self.n_steps)
            arr   = np.interp(x_new, x_old, arr)
        self.result_array = arr;  self.accept()


# =====================================================================
# PARAMETER ROW WIDGET
# =====================================================================
class ParameterRow(QWidget):
    MODES = ['Fixed', 'Sweep', 'Design']

    def __init__(self, label, pkey,
                 default_fixed=0.0, default_from=1.8, default_to=1.2,
                 removable=False, get_nsteps=None, default_mode='Fixed', parent=None):
        super().__init__(parent)
        self.pkey          = pkey
        self.removable     = removable
        self._get_nsteps   = get_nsteps or (lambda: 300)
        self._design_array = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1);  lay.setSpacing(5)

        lbl = QLabel(label);  lbl.setFixedWidth(155)
        lbl.setStyleSheet('color:#c8d0e7;font-size:13px;')
        lay.addWidget(lbl)

        self.mode_combo = QComboBox();  self.mode_combo.addItems(self.MODES)
        self.mode_combo.setFixedWidth(88)
        self.mode_combo.currentTextChanged.connect(self._on_mode)
        lay.addWidget(self.mode_combo)

        self.stack = QStackedWidget();  self.stack.setFixedHeight(30)

        # page 0: Fixed
        p0 = QWidget();  l0 = QHBoxLayout(p0)
        l0.setContentsMargins(0,0,0,0);  l0.setSpacing(4)
        self.spn_fixed = QDoubleSpinBox()
        self.spn_fixed.setDecimals(4);  self.spn_fixed.setRange(-200, 200)
        self.spn_fixed.setValue(default_fixed);  self.spn_fixed.setSingleStep(0.1)
        self.spn_fixed.setFixedWidth(95)
        l0.addWidget(self.spn_fixed);  l0.addStretch()

        # page 1: Sweep
        p1 = QWidget();  l1 = QHBoxLayout(p1)
        l1.setContentsMargins(0,0,0,0);  l1.setSpacing(4)
        self.spn_from = QDoubleSpinBox()
        self.spn_from.setDecimals(4);  self.spn_from.setRange(-200, 200)
        self.spn_from.setValue(default_from);  self.spn_from.setFixedWidth(90)
        arr_lbl = QLabel('→');  arr_lbl.setStyleSheet('color:#4a5270;font-size:13px;')
        self.spn_to = QDoubleSpinBox()
        self.spn_to.setDecimals(4);  self.spn_to.setRange(-200, 200)
        self.spn_to.setValue(default_to);  self.spn_to.setFixedWidth(90)
        l1.addWidget(self.spn_from);  l1.addWidget(arr_lbl)
        l1.addWidget(self.spn_to);    l1.addStretch()

        # page 2: Design
        p2 = QWidget();  l2 = QHBoxLayout(p2)
        l2.setContentsMargins(0,0,0,0);  l2.setSpacing(6)
        self.btn_design = QPushButton('Edit array...')
        self.btn_design.setFixedHeight(26);  self.btn_design.setMinimumWidth(110)
        self.lbl_design = QLabel('not set')
        self.lbl_design.setStyleSheet('color:#4a5270;font-size:12px;')
        self.btn_design.clicked.connect(self._open_design)
        l2.addWidget(self.btn_design);  l2.addWidget(self.lbl_design);  l2.addStretch()

        self.stack.addWidget(p0);  self.stack.addWidget(p1);  self.stack.addWidget(p2)
        lay.addWidget(self.stack, stretch=1)

        if removable:
            self.btn_rm = QPushButton('×')
            self.btn_rm.setFixedSize(24, 24)
            self.btn_rm.setStyleSheet(
                'color:#ff4a6e;font-weight:bold;border:none;background:transparent;font-size:15px;')
            lay.addWidget(self.btn_rm)

        # apply default mode (triggers _on_mode which switches stack page)
        if default_mode != 'Fixed':
            self.mode_combo.setCurrentText(default_mode)

    def _on_mode(self, mode):
        self.stack.setCurrentIndex(self.MODES.index(mode))

    def _open_design(self):
        n   = self._get_nsteps()
        dlg = DesignArrayDialog(n, self.pkey, self._design_array, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._design_array = dlg.result_array
            self.lbl_design.setText(f'{len(self._design_array)} values set')
            self.lbl_design.setStyleSheet('color:#a8ff78;font-size:12px;')

    def get_array(self, n_steps) -> np.ndarray:
        mode = self.mode_combo.currentText()
        if mode == 'Fixed':
            return np.full(n_steps, self.spn_fixed.value())
        elif mode == 'Sweep':
            return np.linspace(self.spn_from.value(), self.spn_to.value(), n_steps)
        else:
            if self._design_array is None:
                return np.zeros(n_steps)
            if len(self._design_array) != n_steps:
                x_old = np.linspace(0, 1, len(self._design_array))
                x_new = np.linspace(0, 1, n_steps)
                return np.interp(x_new, x_old, self._design_array)
            return self._design_array.copy()

    def is_active(self):
        return self.mode_combo.currentText() != 'Fixed'


# =====================================================================
# NONLINEAR WINDOW
# =====================================================================
# =====================================================================
# SLOW TIME WINDOW
# =====================================================================
class SlowTimeWindow(QMainWindow):
    """
    Takes the field at a chosen sweep step, runs NIter_extended more
    iterations saving every single one, then shows 4 heatmaps:
      |a_T|²        — output ring     (x=slow-time iteration, y=θ index)
      |a_T|² summed — all rings       (x=slow-time iteration, y=θ index)
      10·log|a_W|²  — output ring     (x=slow-time iteration, y=FSR μ)
      10·log|a_W|² summed — all rings (x=slow-time iteration, y=FSR μ)
    """
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f'Slow Time Analysis  —  seeded from sweep step {params["step_idx"]}')
        self.setMinimumSize(1600, 840)
        self.setStyleSheet(STYLE)

        self._p          = params
        self._q          = queue.Queue()
        self._stop       = threading.Event()
        self._plotting   = False   # flag for interrupting _draw_all
        # Flag: when True, _poll's done handler will auto-invoke the save
        # dialog after plotting completes. Set by _on_run_save_final.
        self._save_after_run = False
        # full history arrays built after run: (NLattice, NumFSRs, NIter_ext)
        self._hist       = None
        self._poll_timer = QTimer(self); self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        p = self._p
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # Title
        psi_str = f'  ψ={p["psi"]/np.pi:.4f}π' if p.get('psi', 0.0) != 0.0 else ''
        t = QLabel(f'SLOW TIME ANALYSIS  —  seeded from sweep step {p["step_idx"]}  '
                   f'|  Δ={p["detuning"]:.5f}  F={p["F"]:.4f}{psi_str}')
        t.setFont(QFont('Courier New', 11, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(t)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;'); root.addWidget(sep)

        # ── 8 independent figures in a 2×4 QGridLayout ───────────────────
        fig_grid = QWidget()
        gl = QGridLayout(fig_grid)
        gl.setSpacing(4); gl.setContentsMargins(0, 0, 0, 0)
        for r in range(2): gl.setRowStretch(r, 1)
        for c in range(4): gl.setColumnStretch(c, 1)

        def _make_fig(rows=1, cols=1, projection=None):
            fig = Figure(facecolor=DARK_BG, tight_layout=False)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if projection:
                ax = fig.add_subplot(111, projection=projection)
            elif rows == 1 and cols == 1:
                ax = fig.add_subplot(111)
            else:
                ax = None
            return fig, canvas, ax

        # [0,0] Sweep indicator — 2×2 subplots
        self.fig_ind = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_ind = FigureCanvas(self.fig_ind)
        self.canvas_ind.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs0 = self.fig_ind.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
              left=0.10, right=0.97, top=0.95, bottom=0.12)
        ax_pp  = self.fig_ind.add_subplot(gs0[0, 0])
        ax_ppa = self.fig_ind.add_subplot(gs0[0, 1])
        ax_sp  = self.fig_ind.add_subplot(gs0[1, 0])
        ax_spa = self.fig_ind.add_subplot(gs0[1, 1])
        self.ax_ind = [ax_pp, ax_ppa, ax_sp, ax_spa]
        gl.addWidget(self.canvas_ind, 0, 0)

        # [0,1] 4 heatmaps
        self.fig_heat4 = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_heat4 = FigureCanvas(self.fig_heat4)
        self.canvas_heat4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs1 = self.fig_heat4.add_gridspec(2, 2, hspace=0.50, wspace=0.35,
              left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_aT_ring = self.fig_heat4.add_subplot(gs1[0, 0])
        self.ax_aT_all  = self.fig_heat4.add_subplot(gs1[0, 1])
        self.ax_aW_ring = self.fig_heat4.add_subplot(gs1[1, 0])
        self.ax_aW_all  = self.fig_heat4.add_subplot(gs1[1, 1])
        gl.addWidget(self.canvas_heat4, 0, 1)

        # [0,2] Comb spectrum
        self.fig_comb = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_comb = FigureCanvas(self.fig_comb)
        self.canvas_comb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs2 = self.fig_comb.add_gridspec(2, 1, hspace=0.55,
              left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_comb_dB   = self.fig_comb.add_subplot(gs2[0])
        self.ax_comb_zoom = self.fig_comb.add_subplot(gs2[1])
        self.canvas_comb.mpl_connect('button_press_event', self._on_comb_click)
        gl.addWidget(self.canvas_comb, 0, 2)

        # [0,3] 2D Power
        self.fig_2dpow = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_2dpow = FigureCanvas(self.fig_2dpow)
        self.canvas_2dpow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_2dpow.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_2d = self.fig_2dpow.add_subplot(111)
        self.ax_vis_2d.set_facecolor('black')
        self.ax_vis_2d.set_xticks([]); self.ax_vis_2d.set_yticks([])
        gl.addWidget(self.canvas_2dpow, 0, 3)

        # [1,0] ESA + oscilloscope
        self.fig_esa = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_esa = FigureCanvas(self.fig_esa)
        self.canvas_esa.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs3 = self.fig_esa.add_gridspec(3, 1, hspace=0.65,
              left=0.12, right=0.97, top=0.95, bottom=0.10)
        self.ax_esa_dB  = self.fig_esa.add_subplot(gs3[0])
        self.ax_esa_lin = self.fig_esa.add_subplot(gs3[1])
        self.ax_osc     = self.fig_esa.add_subplot(gs3[2])
        gl.addWidget(self.canvas_esa, 1, 0)

        # [1,1] 2D spectrum + y-cuts
        self.fig_spec2d = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_spec2d = FigureCanvas(self.fig_spec2d)
        self.canvas_spec2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs4 = self.fig_spec2d.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
              height_ratios=[1.8, 1], left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_hmap     = self.fig_spec2d.add_subplot(gs4[0, :])
        self.ax_ycut_pwr = self.fig_spec2d.add_subplot(gs4[1, 0])
        self.ax_ycut_phs = self.fig_spec2d.add_subplot(gs4[1, 1])
        self._cbar = None; self.ax_cbar = None
        gl.addWidget(self.canvas_spec2d, 1, 1)

        # [1,2] Photon Flow
        self.fig_flow = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_flow = FigureCanvas(self.fig_flow)
        self.canvas_flow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_flow.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_flow = self.fig_flow.add_subplot(111)
        self.ax_vis_flow.set_facecolor('black')
        self.ax_vis_flow.set_xticks([]); self.ax_vis_flow.set_yticks([])
        gl.addWidget(self.canvas_flow, 1, 2)

        # [1,3] 3D Power
        self.fig_3d = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_3d.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_vis_3d.set_facecolor('black')
        gl.addWidget(self.canvas_3d, 1, 3)

        # Style shared axes
        for ax, col, ttl, yl in [
            (self.ax_aT_ring, '#a8ff78', '|a_T|²  ring',      'θ'),
            (self.ax_aT_all,  '#a8ff78', '|a_T|²  all (sum)', 'θ'),
            (self.ax_aW_ring, '#ff8c42', 'log|a_W|²  ring',   'FSR μ'),
            (self.ax_aW_all,  '#ff8c42', 'log|a_W|²  all',    'FSR μ'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

        for ax, col, ttl in [
            (self.ax_comb_dB,   '#c8d0e7', 'Comb slow-time spectrum (dB)  — click to select tooth'),
            (self.ax_comb_zoom, '#4a9eff', 'Zoomed spectrum — selected tooth (dB)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel('Frequency  (FSR = ? J)', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        for ax, col, ttl, xl, yl in [
            (self.ax_esa_dB,  '#a8ff78', 'ESA spectrum (dB)',          'Angular freq (J)', 'Power (dB)'),
            (self.ax_esa_lin, '#a8ff78', 'ESA spectrum (linear norm.)', 'Angular freq (J)', 'Norm. power'),
            (self.ax_osc,     '#c8d0e7', 'Total power vs slow time',   'Iteration',        'Norm. power'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_hmap.set_facecolor(PANEL_BG)
        self.ax_hmap.set_title('2D slow-time spectrum  (all modes × freq)', color='#ff8c42', fontsize=8, pad=2)
        self.ax_hmap.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.set_ylabel('Mode index μ', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_hmap.spines.values(): sp.set_edgecolor('#3a4560')
        for ax, col, ttl, xl, yl in [
            (self.ax_ycut_pwr, '#c8d0e7', 'Y-cut power  (freq=0)',  'Mode μ', 'Power (dB)'),
            (self.ax_ycut_phs, '#c8d0e7', 'Y-cut phase  (freq=0)',  'Mode μ', 'Phase (rad)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        # Sweep indicator static plots (drawn once)
        _pump_col = '#4a9eff';  _side_col = '#ff4a6e'
        _step_idx = p['step_idx']
        def _norm(arr):
            if arr is None or len(arr) == 0: return np.zeros(1)
            mx = float(np.max(arr)) or 1.
            return np.asarray(arr, float) / mx
        for ax, arr, col, ttl in [
            (ax_pp,  p.get('sweep_pump_out'), _pump_col, 'Pump power  (output ring)'),
            (ax_ppa, p.get('sweep_pump_all'), _pump_col, 'Pump power  (all rings)'),
            (ax_sp,  p.get('sweep_side_out'), _side_col, 'Comb power excl. pump  (output ring)'),
            (ax_spa, p.get('sweep_side_all'), _side_col, 'Comb power excl. pump  (all rings)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            y = _norm(arr)
            ax.plot(y, color=col, lw=0.9)
            ax.axvline(_step_idx, color='white', lw=0.8, ls='--', alpha=0.7)
            ax.set_xlim(0, max(len(y) - 1, 1)); ax.set_ylim(0, 1.05)
            ax.set_title(ttl, color=col, fontsize=7, pad=2)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=6)
            ax.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=6)
            ax.tick_params(colors=TEXT_COL, labelsize=5)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.3, alpha=0.4)

        root.addWidget(fig_grid, stretch=1)

        # ── Controls ──────────────────────────────────────────────────
        ctrl = QWidget(); cl = QHBoxLayout(ctrl)
        cl.setSpacing(8); cl.setContentsMargins(0, 0, 0, 0)

        # Inherited params
        gi = QGroupBox('Inherited parameters'); gi.setMaximumWidth(500)
        il = QGridLayout(gi); il.setSpacing(4)
        def _ro(txt, col='#c8d0e7'):
            lb = QLabel(txt); lb.setStyleSheet(f'color:{col};font-size:12px;'); return lb
        il.addWidget(_ro(f'Δ = {p["detuning"]:.6f} J'),   0, 0)
        il.addWidget(_ro(f'F = {p["F"]:.4f}'),             0, 1)
        il.addWidget(_ro(f'kin = {p["kin"]:.4f}'),         0, 2)
        il.addWidget(_ro(f'kex = {p["kex"]:.4f}'),         1, 0)
        il.addWidget(_ro(f'D2 = {p["D2"]:.4f}'),           1, 1)
        il.addWidget(_ro(f'dt = {p["TStep"]:.3f} J⁻¹'),   1, 2)
        if p.get('psi'):
            il.addWidget(_ro(f'ψ = {p["psi"]/np.pi:.4f} π'), 2, 0)
        for ci, (k, v) in enumerate(p.get('heaters', {}).items()):
            il.addWidget(_ro(f'H{k.split("_")[-1]}={v:+.3f}J', '#ffcc00'), 2, ci)
        cl.addWidget(gi)

        # Run controls
        gr = QGroupBox('Run'); gr.setFixedWidth(380)
        rl = QGridLayout(gr); rl.setSpacing(4)

        def _lbl(txt):
            l = QLabel(txt); l.setStyleSheet('color:#4a5270;font-size:12px;'); return l

        self.spn_niter = QSpinBox()
        self.spn_niter.setRange(2000, 500000); self.spn_niter.setValue(p['NIter_more'])
        self.spn_niter.setSingleStep(1000)

        # Incremental-run state.
        # Each click of "Run round #k" evolves the field by NIter MORE steps,
        # starting from wherever the previous round left off, and overwrites
        # self._hist with the new NIter snapshots (discarding previous rounds'
        # frames so memory usage stays flat at ~one-round-worth).
        # "Run & save final round" does the same plus saves an npz that notes
        # how many rounds were run total (→ total elapsed steps since start).
        self._round_count    = 0      # rounds completed so far
        self._round_last_psi = None   # final field of previous round, used to seed next
        # Per-round NIter list. Appended on every successful completion.
        # Lets us reconstruct the global time axis even when NIter varies
        # between rounds. Total evolved steps = sum(self._round_niters).
        self._round_niters   = []

        self.btn_run  = QPushButton('▶  Run round #1')
        self.btn_run.setFixedHeight(28)
        self.btn_run.setStyleSheet(
            'QPushButton:enabled{color:#a8ff78;border-color:#a8ff78;}'
            'QPushButton:disabled{color:#4a5270;border-color:#2a3050;background:#141a28;}')

        self.btn_stop = QPushButton('■  Stop');  self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False)

        self.btn_save_final = QPushButton('💾  Run & save final round')
        self.btn_save_final.setFixedHeight(28)
        self.btn_save_final.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#4a5270;border-color:#2a3050;background:#141a28;}')

        # Legacy "save only" button — still useful to re-export figures without
        # running another round. Enabled after any round completes.
        self.btn_save = QPushButton('💾  Save current')
        self.btn_save.setFixedHeight(28)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')

        self.prog     = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.lbl_eta  = QLabel(''); self.lbl_eta.setStyleSheet('color:#4a5270;font-size:12px;')

        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(lambda: self._stop.set())
        self.btn_save_final.clicked.connect(self._on_run_save_final)
        self.btn_save.clicked.connect(self._on_save)

        self.btn_lyapunov = QPushButton('🔬  Stability Analysis')
        self.btn_lyapunov.setFixedHeight(28)
        self.btn_lyapunov.setEnabled(False)
        self.btn_lyapunov.setStyleSheet(
            'QPushButton:enabled{color:#ff8c42;border-color:#ff8c42;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_lyapunov.clicked.connect(self._on_stability)

        rl.addWidget(_lbl('NIter (≥2000):'),    0, 0); rl.addWidget(self.spn_niter,     0, 1, 1, 3)
        rl.addWidget(self.btn_run,              1, 0, 1, 2); rl.addWidget(self.btn_stop,       1, 2, 1, 2)
        rl.addWidget(self.btn_save_final,       2, 0, 1, 2); rl.addWidget(self.btn_save,       2, 2, 1, 2)
        rl.addWidget(self.prog,                 3, 0, 1, 3); rl.addWidget(self.lbl_eta,        3, 3)
        rl.addWidget(self.btn_lyapunov,         4, 0, 1, 4)

        # ── [EXPERIMENTAL] Modal decomposition button ──────────────────
        self.btn_modal = QPushButton('🧮  Modal Decomposition')
        self.btn_modal.setFixedHeight(28)
        self.btn_modal.setEnabled(False)
        self.btn_modal.setStyleSheet(
            'QPushButton:enabled{color:#cc88ff;border-color:#cc88ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_modal.clicked.connect(self._on_modal)
        rl.addWidget(self.btn_modal,      5, 0, 1, 4)
        cl.addWidget(gr)

        # Analysis parameters (set before running)
        ga = QGroupBox('Analysis Parameters'); ga.setFixedWidth(420)
        al = QGridLayout(ga); al.setSpacing(4)

        self.spn_fsr_shift = QDoubleSpinBox()
        self.spn_fsr_shift.setRange(0.001, 1000.0); self.spn_fsr_shift.setValue(30.0)
        self.spn_fsr_shift.setSingleStep(1.0); self.spn_fsr_shift.setDecimals(3)
        self.spn_fsr_shift.setToolTip('FSR of a single ring in units of J (shift_factor)')
        self.spn_fsr_shift.valueChanged.connect(lambda _: self._replot_all())

        self.spn_mode_idx = QSpinBox()
        self.spn_mode_idx.setRange(-(p['one_side_fsr']), p['one_side_fsr'])
        self.spn_mode_idx.setValue(4)
        self.spn_mode_idx.setToolTip('Mode index relative to pump to zoom into')
        self.spn_mode_idx.valueChanged.connect(lambda _: self._replot_all())

        self.spn_nest_range = QDoubleSpinBox()
        self.spn_nest_range.setRange(0.01, 100.0); self.spn_nest_range.setValue(2.0)
        self.spn_nest_range.setSingleStep(0.5); self.spn_nest_range.setDecimals(2)
        self.spn_nest_range.setToolTip('Frequency zoom range ±X J for zoomed spectrum and heatmap')
        self.spn_nest_range.valueChanged.connect(lambda _: self._replot_zoom_and_hmap())

        self.spn_ycut_freq = QDoubleSpinBox()
        self.spn_ycut_freq.setRange(-100.0, 100.0); self.spn_ycut_freq.setValue(0.0)
        self.spn_ycut_freq.setSingleStep(0.05); self.spn_ycut_freq.setDecimals(3)
        self.spn_ycut_freq.setToolTip('Frequency at which to take the Y-cut through the 2D heatmap')
        self.spn_ycut_freq.valueChanged.connect(lambda _: self._replot_ycut())

        self.btn_replot = QPushButton('⟳  Replot')
        self.btn_replot.setFixedHeight(28)
        self.btn_replot.setStyleSheet(
            'QPushButton{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:hover{background:#1e2a40;}')
        self.btn_replot.clicked.connect(self._replot_all)

        al.addWidget(_lbl('FSR / J (shift):'),  0, 0); al.addWidget(self.spn_fsr_shift,  0, 1)
        al.addWidget(_lbl('Zoom mode index:'),   0, 2); al.addWidget(self.spn_mode_idx,   0, 3)
        al.addWidget(_lbl('Nest range ±J:'),     1, 0); al.addWidget(self.spn_nest_range, 1, 1)
        al.addWidget(_lbl('Y-cut freq (J):'),    1, 2); al.addWidget(self.spn_ycut_freq,  1, 3)
        al.addWidget(self.btn_replot,            2, 0, 1, 4)
        cl.addWidget(ga)

        # Visualization controls
        gv = QGroupBox('Visualization'); gv.setFixedWidth(340)
        vl = QGridLayout(gv); vl.setSpacing(4)

        self.spn_vis_step = QSpinBox()
        self.spn_vis_step.setRange(0, 99999); self.spn_vis_step.setValue(0)
        self.spn_vis_step.setToolTip('Slow-time iteration index to visualize')

        self.btn_vis_plot  = QPushButton('▶  Plot');   self.btn_vis_plot.setFixedHeight(26)
        self.btn_vis_save  = QPushButton('💾  Save');  self.btn_vis_save.setFixedHeight(26)
        self.btn_vis_gif   = QPushButton('🎞  GIF');   self.btn_vis_gif.setFixedHeight(26)
        self.btn_vis_mp4   = QPushButton('🎬  MP4');   self.btn_vis_mp4.setFixedHeight(26)
        self.lbl_vis_status = QLabel('')
        self.lbl_vis_status.setStyleSheet('color:#4a5270;font-size:11px;')

        self.btn_vis_plot.clicked.connect(self._vis_plot)
        self.btn_vis_save.clicked.connect(self._vis_save_step)
        self.btn_vis_gif.clicked.connect(self._vis_export_gif)
        self.btn_vis_mp4.clicked.connect(self._vis_export_mp4)

        vl.addWidget(_lbl('Step:'),        0, 0); vl.addWidget(self.spn_vis_step,  0, 1, 1, 3)
        vl.addWidget(self.btn_vis_plot,    1, 0, 1, 2); vl.addWidget(self.btn_vis_save, 1, 2, 1, 2)
        vl.addWidget(self.btn_vis_gif,     2, 0, 1, 2); vl.addWidget(self.btn_vis_mp4,  2, 2, 1, 2)
        vl.addWidget(self.lbl_vis_status,  3, 0, 1, 4)
        cl.addWidget(gv)

        # Exit / navigation panel
        ge = QGroupBox('Exit'); ge.setFixedWidth(160)
        ge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        el = QGridLayout(ge); el.setSpacing(4)
        el.setRowStretch(0, 1); el.setRowStretch(2, 1)

        self.btn_vis_exit = QPushButton('✕  Exit')
        self.btn_vis_exit.setStyleSheet('color:#ff4a6e;border-color:#ff4a6e;')
        self.btn_vis_exit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_vis_prev = QPushButton('◀ Prev')
        self.btn_vis_prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_vis_next = QPushButton('Next ▶')
        self.btn_vis_next.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Step-size spinbox — controls how many parent-sweep steps Prev/Next jump
        # over when navigating. Default 5 (a reasonable coarse scan). Max is
        # clamped by parent history length at nav time.
        self.spn_vis_nav_step = QSpinBox()
        self.spn_vis_nav_step.setRange(1, 100000)
        self.spn_vis_nav_step.setValue(10)
        self.spn_vis_nav_step.setToolTip('How many parent-sweep steps Prev/Next jump')
        _lbl_step = QLabel('Step:')
        _lbl_step.setStyleSheet('color:#4a5270;font-size:12px;')
        _lbl_step.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.btn_vis_exit.clicked.connect(self._on_exit)
        self.btn_vis_prev.clicked.connect(lambda: self._vis_nav(-1))
        self.btn_vis_next.clicked.connect(lambda: self._vis_nav(+1))

        el.addWidget(self.btn_vis_exit,     0, 0, 1, 2)
        el.addWidget(_lbl_step,             1, 0)
        el.addWidget(self.spn_vis_nav_step, 1, 1)
        el.addWidget(self.btn_vis_prev,     2, 0)
        el.addWidget(self.btn_vis_next,     2, 1)
        cl.addWidget(ge)

        cl.addStretch()
        root.addWidget(ctrl)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage('Ready — press Run to start slow-time integration')

    # ------------------------------------------------------------------
    def _on_run_save_final(self):
        """Same as _on_run but marks this round as 'final' — auto-invokes save
        dialog when the round completes."""
        self._save_after_run = True
        self._on_run()

    # ------------------------------------------------------------------
    def _on_run(self):
        import time
        p      = self._p
        niter  = self.spn_niter.value()
        # save_every is now hardcoded to 1. We drop the UI spinbox so the
        # control panel has more room, and because users routinely forgot to
        # adjust it for long runs. If you need to downsample, use incremental
        # rounds (each round is short enough to fit in RAM at every=1).
        every  = 1
        n_snap = niter // every

        # Round bookkeeping — _on_run is reused for every round
        this_round = self._round_count + 1
        # Stash the NIter used for THIS round so _poll can commit it to
        # self._round_niters on successful completion. Reading from the
        # spinbox again in _poll would be wrong if the user changed the
        # value during the run.
        self._this_round_niter = int(niter)

        self.btn_run.setEnabled(False)
        self.btn_save_final.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_lyapunov.setEnabled(False)
        self.btn_modal.setEnabled(False)
        self.prog.setValue(0); self._stop.clear(); self._hist = None

        DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
            p['NLattice'], p['one_side_fsr'], p['D2'],
            p['kin'], p['kex'], p['ISite'], p['OSite'])

        # For cylinder lattice: always rebuild H with the correct swept ψ at this step.
        psi_val = p.get('psi', 0.0)
        ls      = p.get('linear_state', {})
        if ls.get('is_cyl') and build_hamiltonian is not None:
            H_full = build_hamiltonian(
                ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2), ls.get('phi_iqh1', np.pi/2),
                ls.get('phi_aqh0', np.pi/4), ls.get('phi_aqh1', np.pi/4), psi_val)
            H = H_full[1:, 1:].copy()
        else:
            H = p['H_mat'].copy()
        for k, v in p.get('heaters', {}).items():
            n = int(k.split('_')[1]); H[n-1, n-1] += v

        # Log all parameters for verification
        h_trace = f'H[0,0]={H[0,0]:.4f}' + (f'  H[1,0]={H[1,0]:.4f}' if H.shape[0] > 1 else '')
        self.status.showMessage(
            f'Running round #{this_round} — Δ={p["detuning"]:.5f}  F={p["F"]:.4f}  '
            f'ψ={psi_val/np.pi:.4f}π  kin={p["kin"]:.4f}  kex={p["kex"]:.4f}  '
            f'D2={p["D2"]:.4f}  dt={p["TStep"]:.3f}  {h_trace}')

        props, fc_arr, _ = build_propagators(
            H, DispMat, LM, p['detuning'], p['TStep'],
            p['F'], PumpFSR, p['ISite'])
        props_j = jnp.array(props); fc_j = jnp.array(fc_arr)
        # run 1 iteration at a time so we can save every snapshot
        stepper1 = get_stepper(1)

        # Seed field: first round starts from the snapshot passed in from the
        # parent window; subsequent rounds pick up from where the previous
        # round left off (stored in self._round_last_psi).
        if self._round_count == 0 or self._round_last_psi is None:
            a_T = jnp.array(p['a_T_init'])
        else:
            a_T = jnp.array(self._round_last_psi)
        _q  = self._q

        def worker():
            nonlocal a_T
            # pre-allocate history: (NLattice, NumFSRs, n_snap)
            NL, NF = p['NLattice'], NumFSRs
            hist = np.empty((NL, NF, n_snap), dtype=np.complex128)
            t0   = time.time()
            snap = 0
            try:
                for i in range(niter):
                    if self._stop.is_set():
                        _q.put(('stopped', hist[:, :, :snap], snap)); return
                    a_T = stepper1(a_T, props_j, fc_j, p['TStep'])
                    if (i + 1) % every == 0:
                        hist[:, :, snap] = np.array(a_T)
                        snap += 1
                    if i % max(1, niter // 200) == 0:
                        elapsed = time.time() - t0
                        _q.put(('progress', i + 1, niter, snap, elapsed))
                _q.put(('done', hist, snap))
            except Exception as exc:
                import traceback; traceback.print_exc()
                _q.put(('stopped', hist[:, :, :snap], snap))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_timer.start()

    # ------------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg[0]
                if kind == 'progress':
                    _, done, total, snap, elapsed = msg
                    pct = int(100 * done / total)
                    self.prog.setValue(pct)
                    rem = elapsed / done * (total - done) if done else 0
                    em, es = divmod(int(rem), 60)
                    self.lbl_eta.setText(f'{em}m{es:02d}s')
                    self.status.showMessage(
                        f'Round #{self._round_count + 1}: iter {done}/{total}  —  '
                        f'{snap} snapshots  —  ETA {em}m{es:02d}s')
                elif kind in ('done', 'stopped'):
                    _, hist, snap = msg
                    self._poll_timer.stop()
                    self.btn_stop.setEnabled(False)
                    self.prog.setValue(100 if kind == 'done' else self.prog.value())
                    if kind == 'done' and snap > 0:
                        # Round completed successfully — commit it, including
                        # the NIter actually used (not whatever the spinbox
                        # currently shows — user may have changed it mid-run).
                        self._round_count += 1
                        self._round_last_psi = np.asarray(hist[:, :, -1]).copy()
                        self._round_niters.append(int(self._this_round_niter))
                    # Update button label to next round number
                    self.btn_run.setText(f'▶  Run round #{self._round_count + 1}')
                    self.btn_run.setEnabled(True)
                    self.btn_save_final.setEnabled(True)
                    total_steps = sum(self._round_niters)
                    if kind == 'done':
                        # Build a compact history string: "10k+2k+2k" if rounds differ,
                        # "3×10k" if they don't. Helps the user track what they've done.
                        if len(set(self._round_niters)) == 1 and len(self._round_niters) > 1:
                            hist_str = f'{len(self._round_niters)}×{self._round_niters[0]:,}'
                        else:
                            hist_str = '+'.join(f'{n:,}' for n in self._round_niters)
                        txt = (f'✓ Round {self._round_count} done — '
                               f'{snap} snapshots  |  rounds: {hist_str}  |  '
                               f'total evolved: {total_steps:,} steps')
                    else:
                        txt = f'Stopped mid-round — {snap} snapshots from incomplete round.'
                    self.status.showMessage(txt)
                    if snap > 0:
                        self._hist = hist[:, :, :snap]
                        self.spn_vis_step.setRange(0, snap - 1)
                        self.spn_vis_step.setValue(snap - 1)
                        self.btn_lyapunov.setEnabled(True)
                        self.btn_modal.setEnabled(True)
                        self.btn_save.setEnabled(True)
                        self.status.showMessage('Plotting…')
                        QApplication.processEvents()
                        self._draw_all()
                        self.status.showMessage(txt)
                        # If this run was initiated by "Run & save final round",
                        # kick off the save dialog now that plotting is done.
                        if getattr(self, '_save_after_run', False):
                            self._save_after_run = False
                            if kind == 'done':
                                self._on_save()
        except queue.Empty:
            pass

    def _on_stability(self):
        """Open Stability Analysis window (Lyapunov + Jacobian tabs)."""
        if self._hist is None:
            QMessageBox.warning(self, 'No data', 'Run simulation first.'); return
        try:
            self._stability_win = StabilityWindow(self._p, self._hist, parent=self)
            self._stability_win.show()
        except Exception as exc:
            import traceback
            msg = traceback.format_exc()
            print('STABILITY WINDOW ERROR:\n' + msg)
            QMessageBox.critical(self, 'Stability Analysis Error',
                                 f'{type(exc).__name__}: {exc}\n\nSee console for full traceback.')

    def _on_lyapunov(self):
        """Open Lyapunov Analysis dialog, run FTLE, show results."""
        if self._hist is None:
            QMessageBox.warning(self, 'No data', 'Run simulation first.'); return

        p = self._p

        # ── Parameter dialog ──────────────────────────────────────────
        dlg = QDialog(self); dlg.setWindowTitle('Lyapunov Analysis Parameters')
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setStyleSheet(STYLE); dlg.setMinimumWidth(340)
        fl = QGridLayout(dlg); fl.setSpacing(8)

        def _lbl(t):
            l = QLabel(t); l.setStyleSheet('color:#c8d0e7;font-size:13px;'); return l

        spn_T   = QSpinBox();        spn_T.setRange(100, 500000); spn_T.setValue(500000); spn_T.setSingleStep(500)
        spn_K   = QSpinBox();        spn_K.setRange(1, 1000);     spn_K.setValue(20);    spn_K.setSingleStep(5)
        spn_tr  = QSpinBox();        spn_tr.setRange(0, 1000);    spn_tr.setValue(50);   spn_tr.setSingleStep(10)
        spn_eps = QDoubleSpinBox();  spn_eps.setDecimals(2);      spn_eps.setRange(-20, -1); spn_eps.setValue(-10)
        spn_eps.setToolTip('Perturbation size = 10^(this value)')

        fl.addWidget(_lbl('Total steps T:'),          0, 0); fl.addWidget(spn_T,   0, 1)
        fl.addWidget(_lbl('Renorm. interval K:'),     1, 0); fl.addWidget(spn_K,   1, 1)
        fl.addWidget(_lbl('Transient cycles:'),       2, 0); fl.addWidget(spn_tr,  2, 1)
        fl.addWidget(_lbl('log₁₀(ε):'),              3, 0); fl.addWidget(spn_eps, 3, 1)

        def _show_help():
            h = QDialog(dlg); h.setWindowTitle('FTLE — Theory & Parameters')
            h.setWindowFlags(h.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            h.setStyleSheet(STYLE); h.setMinimumSize(640, 700)
            vl = QVBoxLayout(h); vl.setSpacing(8); vl.setContentsMargins(14,14,14,14)

            txt = QPlainTextEdit(); txt.setReadOnly(True)
            txt.setStyleSheet('QPlainTextEdit{background:#0a0c14;color:#c8d0e7;'
                              'font-family:"Courier New";font-size:12px;border:1px solid #1e2230;}')
            txt.setPlainText("""
FINITE-TIME LYAPUNOV EXPONENT (FTLE) ANALYSIS
══════════════════════════════════════════════

WHAT IS IT?
───────────
The FTLE measures the average rate at which two initially nearby
trajectories in phase space diverge (or converge) over a finite
time T. It is the central diagnostic for distinguishing:

  • Stable solutions   →  λ < 0   (perturbations decay)
  • Chaotic solutions  →  λ > 0   (perturbations grow exponentially)
  • Quasi-periodic     →  λ ≈ 0   (marginal, neither)

DERIVATION
──────────
The slow-time LLE evolves the field Ψ ∈ ℂ^(NL×NF) by one map step:

    Ψ(t+1) = F(Ψ(t))

For two nearby trajectories Ψ₁(t) and Ψ₂(t) = Ψ₁(t) + δ(t),
linearising F around Ψ₁ gives:

    δ(t+1) ≈ J(Ψ₁(t)) · δ(t)

where J is the Jacobian of F. After t steps:

    δ(t) = J(t-1) · J(t-2) · … · J(0) · δ(0)

The maximal Lyapunov exponent is:

    λ = lim_{T→∞}  (1/T) · log( ‖δ(T)‖ / ‖δ(0)‖ )

NUMERICAL ALGORITHM (Wolf et al. 1985)
───────────────────────────────────────
Rather than computing J explicitly, we evolve TWO trajectories
and measure their separation directly.

Step 1 — Initialise:
    Ψ₁(0) = last frame of the slow-time history (on attractor)
    δ(0)  = random unit vector × ε     (tiny perturbation)
    Ψ₂(0) = Ψ₁(0) + δ(0)

Step 2 — Transient burn-in (N_trans cycles × K steps):
    Evolve both trajectories and renormalise every K steps,
    but do NOT accumulate log-growth yet. This aligns δ with
    the most unstable direction of the local dynamics.

Step 3 — Accumulate (T/K cycles):
    For each renormalisation cycle i = 1 … T/K:
      a) Evolve Ψ₁ and Ψ₂ forward K slow-time steps.
      b) Measure separation:  dᵢ = ‖Ψ₂ - Ψ₁‖₂
      c) Accumulate:  log_sum += log(dᵢ / ε)
      d) Renormalise:  Ψ₂ ← Ψ₁ + ε · (Ψ₂ - Ψ₁) / dᵢ
         (pulls Ψ₂ back to distance ε from Ψ₁, same direction)

Step 4 — Final estimate:
    λ(T) = log_sum / T  (units: per slow-time step)

The running estimate after cycle i:
    λ(i·K) = (Σⱼ≤ᵢ log(dⱼ/ε)) / (i·K)
converges to a plateau — this is what the left plot shows.

WHY RENORMALISE?
────────────────
Without renormalisation, dᵢ would saturate at the attractor
diameter (chaotic case) or underflow to zero (stable case),
making long-time estimation impossible. Renormalisation keeps
the perturbation in the linear regime (‖δ‖ ≪ attractor scale)
while faithfully accumulating the total log-growth.

PARAMETERS
──────────
T (Total steps):
    Total number of slow-time iterations to integrate after the
    transient. Larger T gives a more converged λ estimate.
    Rule of thumb: T ≫ 1/|λ|. Default: 500,000.

K (Renorm. interval):
    Number of steps between each renormalisation. Smaller K
    renormalises more frequently — better for strongly chaotic
    systems. Larger K reduces overhead. Default: 20.

Transient cycles:
    Number of renormalisation cycles to discard before
    accumulating. Discards initial transient and aligns δ with
    the dominant unstable mode. Default: 50 (= 1000 steps).

log₁₀(ε):
    Size of the initial perturbation δ(0). Must be much smaller
    than the typical field amplitude (~0.01–0.1 in your system)
    to stay in the linear regime. ε = 10⁻¹⁰ is safe. Default: -10.

INTERPRETATION
──────────────
  λ > 0  →  CHAOTIC    — exponential divergence, comb is unstable
  λ < 0  →  STABLE     — perturbations decay, coherent comb state
  λ ≈ 0  →  MARGINAL   — quasi-periodic or limit cycle

The convergence plot shows λ(t) vs renormalisation cycle. A clear
plateau indicates convergence. Oscillations mean T is too short
or the system is near the stability boundary.
""")
            vl.addWidget(txt)
            btn_close = QPushButton('Close'); btn_close.clicked.connect(h.accept)
            br = QHBoxLayout(); br.addStretch(); br.addWidget(btn_close)
            vl.addLayout(br)
            h.exec_()

        btn_help = QPushButton('❓  Help')
        btn_ok   = QPushButton('Run');    btn_ok.clicked.connect(dlg.accept)
        btn_can  = QPushButton('Cancel'); btn_can.clicked.connect(dlg.reject)
        btn_help.clicked.connect(_show_help)
        brow = QHBoxLayout()
        brow.addWidget(btn_help); brow.addStretch()
        brow.addWidget(btn_ok);   brow.addWidget(btn_can)
        fl.addLayout(brow, 4, 0, 1, 2)

        if dlg.exec_() != QDialog.Accepted: return

        T_steps   = spn_T.value()
        K         = spn_K.value()
        N_trans   = spn_tr.value()
        eps       = 10 ** spn_eps.value()

        # ── Run FTLE in a result window ───────────────────────────────
        win = LyapunovWindow(p, self._hist, T_steps, K, N_trans, eps, parent=self)
        win.show()
        win.run()


# =====================================================================
# STABILITY WINDOW  —  tabbed container: Lyapunov | Jacobian
# =====================================================================
class StabilityWindow(QMainWindow):
    def __init__(self, p, hist, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Stability Analysis')
        self.setMinimumSize(820, 600)
        self.setStyleSheet(STYLE)
        self._p    = p
        self._hist = hist

        central = QWidget(); self.setCentralWidget(central)
        root    = QVBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(6)

        title = QLabel('STABILITY ANALYSIS')
        title.setFont(QFont('Courier New', 11, QFont.Bold))
        title.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(title)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            'QTabWidget::pane{border:1px solid #1e2230;background:#08090d;}'
            'QTabBar::tab{background:#0d0f1a;color:#4a5270;padding:6px 18px;border:1px solid #1e2230;}'
            'QTabBar::tab:selected{background:#0a0c14;color:#c8d0e7;border-bottom:2px solid #00e5ff;}')
        root.addWidget(tabs, stretch=1)

        # ── Tab 1: Resolution (grid sampling) ────────────────────────
        # Diagnostic first, because whether the Jacobian/Floquet results are
        # trustworthy depends on whether solitons are resolved enough for the
        # Goldstone to appear near 0 rather than hidden in the bulk at -κ.
        res_tab = QWidget()
        res_lay = QVBoxLayout(res_tab); res_lay.setContentsMargins(8,8,8,8); res_lay.setSpacing(6)
        try:
            self._res_widget = _ResolutionTabWidget(p, hist, parent_win=self)
            res_lay.addWidget(self._res_widget)
        except Exception as exc:
            import traceback; traceback.print_exc()
            res_lay.addWidget(QLabel(f'Resolution tab error: {exc}'))
        tabs.addTab(res_tab, '📐  Sampling check')

        # ── Tab 2: Lyapunov ──────────────────────────────────────────
        lyap_tab = QWidget()
        lyap_lay = QVBoxLayout(lyap_tab); lyap_lay.setContentsMargins(8,8,8,8); lyap_lay.setSpacing(6)
        try:
            self._lyap_widget = _LyapunovTabWidget(p, hist, parent_win=self)
            lyap_lay.addWidget(self._lyap_widget)
        except Exception as exc:
            import traceback; traceback.print_exc()
            lyap_lay.addWidget(QLabel(f'Lyapunov tab error: {exc}'))
        tabs.addTab(lyap_tab, '📈  Lyapunov  (FTLE)')

        # ── Tab 3: Jacobian ──────────────────────────────────────────
        jac_tab = QWidget()
        jac_lay = QVBoxLayout(jac_tab); jac_lay.setContentsMargins(8,8,8,8); jac_lay.setSpacing(6)
        try:
            self._jac_widget = _JacobianTabWidget(p, hist, parent_win=self)
            jac_lay.addWidget(self._jac_widget)
        except Exception as exc:
            import traceback; traceback.print_exc()
            jac_lay.addWidget(QLabel(f'Jacobian tab error: {exc}'))
        tabs.addTab(jac_tab, '🔢  Jacobian  (eigenspectrum)')

        # ── Tab 4: Monodromy ─────────────────────────────────────────
        mono_tab = QWidget()
        mono_lay = QVBoxLayout(mono_tab); mono_lay.setContentsMargins(8,8,8,8); mono_lay.setSpacing(6)
        try:
            self._mono_widget = _MonodromyTabWidget(p, hist, parent_win=self)
            mono_lay.addWidget(self._mono_widget)
        except Exception as exc:
            import traceback; traceback.print_exc()
            mono_lay.addWidget(QLabel(f'Monodromy tab error: {exc}'))
        tabs.addTab(mono_tab, '🔄  Monodromy  (Floquet)')

    def closeEvent(self, event):
        if hasattr(self, '_res_widget'):   self._res_widget.stop()
        if hasattr(self, '_lyap_widget'):  self._lyap_widget.stop()
        if hasattr(self, '_jac_widget'):   self._jac_widget.stop()
        if hasattr(self, '_mono_widget'):  self._mono_widget.stop()
        super().closeEvent(event)


# =====================================================================
# RESOLUTION TAB  —  soliton-sampling diagnostic
# =====================================================================
class _ResolutionTabWidget(QWidget):
    """First stability tab: inspect how many grid points sample each peak
    in |a_T|². This is diagnostic for whether the grid resolves solitons
    adequately for Jacobian/Floquet analysis to find Goldstone modes.

    Analytical target: FWHM ≥ ~8 grid points. Below 6, Goldstones get
    pushed into the bulk loss cluster at Re(λ) = -κ. Above 12, they
    appear at numerical precision near Re(λ) = 0."""

    def __init__(self, p, hist, parent_win=None):
        super().__init__()
        self._p    = p
        self._hist = hist

        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(6)

        title = QLabel('SAMPLING CHECK')
        title.setFont(QFont('Courier New', 11, QFont.Bold))
        title.setStyleSheet(f'color:{ACCENT};')
        lay.addWidget(title)

        subtitle = QLabel(
            'Counts how many grid points sample each peak in |a_T|². '
            'On a finite grid, continuous θ-translation is only a symmetry if the '
            'profile is resolved well enough to be treated as continuous. With few '
            'samples per peak, θ-translation is numerically broken — Goldstone modes '
            'get lifted off zero and hide on the bulk line at Re(λ) ≈ −κ. '
            'Rough targets: ≥10 grid pts per FWHM is well-resolved, 6–9 is marginal, '
            '<6 is likely to hide Goldstones.')
        subtitle.setStyleSheet('color:#8892b0;font-size:11px;')
        subtitle.setWordWrap(True)
        lay.addWidget(subtitle)

        # ── Control row ──────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        def _lbl(t):
            l = QLabel(t); l.setStyleSheet('color:#c8d0e7;font-size:12px;'); return l
        # Threshold fraction for "FWHM" boundary
        self.spn_thresh = QDoubleSpinBox()
        self.spn_thresh.setDecimals(2); self.spn_thresh.setRange(0.05, 0.95)
        self.spn_thresh.setSingleStep(0.05); self.spn_thresh.setValue(0.50)
        self.spn_thresh.setToolTip('Width is measured at this fraction of each peak intensity (default 0.5 = FWHM)')
        ctrl_row.addWidget(_lbl('peak fraction:')); ctrl_row.addWidget(self.spn_thresh)
        ctrl_row.addStretch()

        self.btn_run = QPushButton('▶  Run sampling analysis')
        self.btn_run.setFixedHeight(26)
        self.btn_run.setStyleSheet('QPushButton{color:#a8ff78;border-color:#a8ff78;}')
        self.btn_run.clicked.connect(self._run)
        ctrl_row.addWidget(self.btn_run)
        lay.addLayout(ctrl_row)

        # ── Plot canvas ──────────────────────────────────────────────
        self.fig    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        self.ax     = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL_BG)
        self.ax.set_xlabel('θ / 2π', color=TEXT_DIM, fontsize=10)
        self.ax.set_ylabel('|a_T|²', color=TEXT_DIM, fontsize=10)
        self.ax.set_title('Press ▶ to analyze grid sampling of the latest state',
                          color=TEXT_COL, fontsize=10, pad=4)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, stretch=1)

        # Navigation toolbar for interactive zoom
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar_res = NavigationToolbar2QT(self.canvas, self)
        self.toolbar_res.setStyleSheet(
            'QToolBar{background:#141a28;border:1px solid #3a4560;spacing:2px;}'
            'QToolButton{background:#1e2538;color:#c8d0e7;border:1px solid #3a4560;'
            '  border-radius:2px;padding:3px;}'
            'QToolButton:hover{background:#2a3148;border-color:#00e5ff;}'
            'QToolButton:checked{background:#00e5ff;color:#0e1520;border-color:#00e5ff;}'
            'QLabel{color:#c8d0e7;font-size:11px;background:#141a28;}')
        lay.addWidget(self.toolbar_res)

        # ── Summary panel ────────────────────────────────────────────
        self.lbl_summary = QLabel('')
        self.lbl_summary.setFont(QFont('Courier New', 10))
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet('color:#c8d0e7;background:#0a0d18;padding:8px;'
                                       'border:1px solid #2a3148;border-radius:3px;')
        self.lbl_summary.setMinimumHeight(80)
        self.lbl_summary.setMaximumHeight(200)
        lay.addWidget(self.lbl_summary)

        # ── Save buttons ─────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addStretch()
        self.btn_save_png = QPushButton('💾  Save PNG')
        self.btn_save_png.setFixedHeight(28)
        self.btn_save_png.setEnabled(False)
        self.btn_save_png.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_png.clicked.connect(lambda: self._save_fig('png'))
        self.btn_save_svg = QPushButton('💾  Save SVG')
        self.btn_save_svg.setFixedHeight(28)
        self.btn_save_svg.setEnabled(False)
        self.btn_save_svg.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_svg.clicked.connect(lambda: self._save_fig('svg'))
        save_row.addWidget(self.btn_save_png)
        save_row.addWidget(self.btn_save_svg)
        lay.addLayout(save_row)

    def _save_fig(self, fmt):
        """Save the sampling-check figure as PNG or SVG."""
        default_dir = ''
        try:
            # Try to reuse the nonlinear simulator's save folder if available
            pw = self.parent()
            while pw is not None and not hasattr(pw, '_get_nl_folder'):
                pw = pw.parent()
            if pw is not None:
                default_dir = pw._get_nl_folder()
        except Exception:
            default_dir = ''
        fname, _ = QFileDialog.getSaveFileName(
            self, f'Save sampling-check {fmt.upper()}',
            f'{default_dir}/sampling_check.{fmt}' if default_dir else f'sampling_check.{fmt}',
            f'{fmt.upper()} (*.{fmt})')
        if not fname: return
        if not fname.lower().endswith(f'.{fmt}'):
            fname += f'.{fmt}'
        try:
            self.fig.savefig(fname, dpi=150, facecolor=self.fig.get_facecolor(),
                             bbox_inches='tight')
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed',
                                 f'Could not save {fmt.upper()}:\n{exc}')

    # ---- no threads needed; analysis is cheap ----
    def stop(self): pass

    def _run(self):
        import numpy as np
        from scipy.signal import find_peaks

        # Use the last snapshot from the passed history
        hist = self._hist
        if hist is None or hist.ndim != 3 or hist.shape[2] == 0:
            self.lbl_summary.setText('No state available.')
            return
        a_T = np.asarray(hist[:, :, -1])
        # Multi-ring: pick the ring with biggest peak
        if a_T.shape[0] > 1:
            ring_peaks = [float(np.abs(a_T[n, :]).max()) for n in range(a_T.shape[0])]
            ring = int(np.argmax(ring_peaks))
        else:
            ring = 0
        I = np.abs(a_T[ring, :])**2
        N = len(I)
        I_max  = float(I.max())
        I_mean = float(I.mean())
        I_min  = float(I.min())

        thresh_frac = float(self.spn_thresh.value())

        # --- peak detection on periodic array ---
        # Prominence threshold: peak must rise at least 20% of (max-mean) above
        # surroundings. Height threshold: peak intensity must be > mean + 20%.
        dyn    = I_max - I_mean
        prom   = max(0.2 * dyn, 1e-12)
        height = I_mean + 0.2 * dyn
        Ipad = np.concatenate([I, I, I])
        pks_padded, _ = find_peaks(Ipad, height=height, prominence=prom)
        pks = sorted({int(p - N) for p in pks_padded if N <= p < 2*N})
        pks = np.array(pks, dtype=int)

        # For each peak, measure width at thresh_frac·peak
        peak_info = []
        for pk in pks:
            pk_I = I[pk]
            th = pk_I * thresh_frac
            left = pk
            for k in range(1, N//2):
                idx = (pk - k) % N
                if I[idx] < th: break
                left = idx
            right = pk
            for k in range(1, N//2):
                idx = (pk + k) % N
                if I[idx] < th: break
                right = idx
            if left <= right:
                npts = right - left + 1
                idx_list = list(range(left, right+1))
            else:
                npts = (N - left) + right + 1
                idx_list = list(range(left, N)) + list(range(0, right+1))
            peak_info.append({
                'idx': int(pk), 'peak_I': float(pk_I),
                'left': int(left), 'right': int(right),
                'npts': int(npts), 'indices': idx_list,
            })
        # Sort peaks by height descending
        peak_info.sort(key=lambda p: -p['peak_I'])

        # --- plot ---
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')

        theta_over_2pi = np.arange(N) / N   # θ/2π ∈ [0, 1)
        # Continuous line (computed by sinc-interpolating to a finer grid via FFT padding)
        # to show what the 'true' continuous profile looks like
        aW = np.fft.fft(a_T[ring, :])
        # zero-pad to 4× resolution
        Nfine = 4 * N
        aW_fine = np.zeros(Nfine, dtype=complex)
        aW_fine[:(N+1)//2] = aW[:(N+1)//2]
        aW_fine[-(N//2):] = aW[-(N//2):]
        a_T_fine = np.fft.ifft(aW_fine) * (Nfine / N)
        I_fine = np.abs(a_T_fine)**2
        theta_fine = np.arange(Nfine) / Nfine

        self.ax.plot(theta_fine, I_fine, color='#4a9eff', lw=1.0, alpha=0.55,
                     label='continuous |a_T|² (4× upsampled)', zorder=1)

        # Grid samples
        is_in_peak = np.zeros(N, dtype=bool)
        peak_color = ['#ffd700', '#ff8c42', '#a8ff78', '#ff4a6e', '#c084fc',
                      '#00e5ff', '#ff94c2', '#ffeb3b', '#81d4fa', '#b2ff59']
        # All grid samples as small grey dots
        self.ax.scatter(theta_over_2pi, I, c='#6a7494', s=14, zorder=2,
                        label=f'all grid samples ({N})')

        # Highlight peak-region samples
        for i, pi in enumerate(peak_info):
            col = peak_color[i % len(peak_color)]
            idx_arr = np.array(pi['indices'])
            self.ax.scatter(theta_over_2pi[idx_arr], I[idx_arr],
                            c=col, s=45, zorder=4,
                            edgecolors='white', linewidths=0.4,
                            label=f'peak {i+1}: {pi["npts"]} pts')
            # Draw threshold line for this peak over the peak region
            pk_idx = pi['idx']
            x_center = theta_over_2pi[pk_idx]
            self.ax.axhline(pi['peak_I']*thresh_frac,
                            color=col, ls=':', lw=0.6, alpha=0.4, zorder=3)

        # Legend — cap at 6 entries max so it doesn't overflow
        handles, labels = self.ax.get_legend_handles_labels()
        if len(handles) > 8:
            handles, labels = handles[:8], labels[:8] + ['(more peaks truncated)']
        self.ax.legend(handles, labels, fontsize=7, frameon=False, labelcolor=TEXT_COL,
                       loc='upper right', ncol=2)

        self.ax.set_xlim(0, 1)
        if I_max > 0:
            self.ax.set_ylim(-0.05*I_max, 1.15*I_max)
        self.ax.set_xlabel('θ / 2π', color=TEXT_DIM, fontsize=10)
        self.ax.set_ylabel('|a_T|²', color=TEXT_DIM, fontsize=10)
        self.ax.set_title(
            f'Grid sampling at {int(100*thresh_frac)}% of each peak  '
            f'(N = {N} total grid points)',
            color=TEXT_COL, fontsize=10, pad=4)
        self.canvas.draw_idle()

        # --- summary text ---
        if len(peak_info) == 0:
            msg = (f'No localized peaks detected. |a_T|²: max={I_max:.4f}, mean={I_mean:.4f}, '
                   f'min={I_min:.4f}. Profile looks extended/turbulent — grid-resolution '
                   f'requirements for Jacobian analysis are less clear; try running the '
                   f'Jacobian to see what the spectrum looks like.')
            self.lbl_summary.setText(msg)
            self.btn_save_png.setEnabled(True)
            self.btn_save_svg.setEnabled(True)
            return

        lines = [
            f'Grid: N = {N} points, spacing dθ = 2π/N = {2*np.pi/N:.4f} rad.',
            f'|a_T|² peak = {I_max:.4f}, mean = {I_mean:.4f}, peak/mean = {I_max/max(I_mean,1e-30):.1f}',
            f'Detected {len(peak_info)} peak(s):',
        ]
        for i, pi in enumerate(peak_info):
            grade = ('under-resolved (Goldstone likely absorbed into bulk)'
                     if pi['npts'] < 6
                     else 'marginal (Goldstone may or may not resolve)'
                     if pi['npts'] < 10
                     else 'well-resolved (Goldstone should appear near 0)')
            lines.append(
                f'  peak {i+1}:  width @ {int(100*thresh_frac)}% = {pi["npts"]} grid pts   [{grade}]')

        # Rule-of-thumb recommendation
        if peak_info:
            min_npts = min(p['npts'] for p in peak_info)
            if min_npts < 8:
                N_rec = int(np.ceil(N * 8 / min_npts))
                if N_rec % 2 == 0: N_rec += 1
                lines.append('')
                lines.append(f'Recommendation: to get ≥ 8 grid points across the narrowest peak, '
                             f'use N_τ ≥ {N_rec} (currently {N}). '
                             f'Increase `one_side_fsr` to ≥ {N_rec//2} before re-running the sweep.')
        self.lbl_summary.setText('\n'.join(lines))
        self.btn_save_png.setEnabled(True)
        self.btn_save_svg.setEnabled(True)


# =====================================================================
# LYAPUNOV TAB  —  inline FTLE widget (no separate window)
# =====================================================================
class _LyapunovTabWidget(QWidget):
    """Inline FTLE tab — all LyapunovWindow content embedded directly."""
    def __init__(self, p, hist, parent_win=None):
        super().__init__()
        self._p       = p
        self._hist    = hist
        self._stop    = threading.Event()
        self._q       = queue.Queue()
        self._timer   = QTimer(self); self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._running_data = []

        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(6)

        # ── Title ────────────────────────────────────────────────────
        self.lbl_title = QLabel('FINITE-TIME LYAPUNOV EXPONENT')
        self.lbl_title.setFont(QFont('Courier New', 11, QFont.Bold))
        self.lbl_title.setStyleSheet(f'color:{ACCENT};')
        lay.addWidget(self.lbl_title)

        # ── Parameter row ────────────────────────────────────────────
        param_row = QHBoxLayout()
        def _lbl(t):
            l = QLabel(t); l.setStyleSheet('color:#c8d0e7;font-size:12px;'); return l

        self.spn_T   = QSpinBox();       self.spn_T.setRange(100,500000); self.spn_T.setValue(500000); self.spn_T.setSingleStep(500)
        self.spn_K   = QSpinBox();       self.spn_K.setRange(1,1000);     self.spn_K.setValue(20);    self.spn_K.setSingleStep(5)
        self.spn_tr  = QSpinBox();       self.spn_tr.setRange(0,1000);    self.spn_tr.setValue(50);   self.spn_tr.setSingleStep(10)
        self.spn_eps = QDoubleSpinBox(); self.spn_eps.setDecimals(2);     self.spn_eps.setRange(-20,-1); self.spn_eps.setValue(-10)
        self.spn_eps.setToolTip('Perturbation size = 10^(this value)')

        for lbl_txt, spn in [('T:', self.spn_T), ('K:', self.spn_K),
                              ('transient:', self.spn_tr), ('log₁₀(ε):', self.spn_eps)]:
            param_row.addWidget(_lbl(lbl_txt)); param_row.addWidget(spn)
        param_row.addStretch()

        self.btn_run = QPushButton('▶  Run FTLE')
        self.btn_run.setFixedHeight(26)
        self.btn_run.setStyleSheet(
            'QPushButton:enabled{color:#a8ff78;border-color:#a8ff78;}'
            'QPushButton:disabled{color:#4a5270;border-color:#2a3050;background:#141a28;}')
        self.btn_run.clicked.connect(self._launch)
        param_row.addWidget(self.btn_run)
        lay.addLayout(param_row)

        # ── Plot canvas ──────────────────────────────────────────────
        self.fig    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL_BG)
        self.ax.set_xlabel('Cycle', color=TEXT_DIM, fontsize=9)
        self.ax.set_ylabel('λ(t)', color=TEXT_DIM, fontsize=9)
        self.ax.set_title('Press ▶ Run FTLE to start', color=TEXT_COL, fontsize=10, pad=4)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')
        lay.addWidget(self.canvas, stretch=1)

        # ── Progress + status ────────────────────────────────────────
        self.prog = QProgressBar(); self.prog.setRange(0,100); self.prog.setValue(0)
        self.lbl_status = QLabel('Ready')
        self.lbl_status.setStyleSheet('color:#4a5270;font-size:12px;')
        lay.addWidget(self.prog)
        lay.addWidget(self.lbl_status)

        # ── Stop + Save ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_stop_lyap = QPushButton('■  Stop')
        self.btn_stop_lyap.setFixedHeight(28)
        self.btn_stop_lyap.clicked.connect(self._stop.set)
        self.btn_save_lyap = QPushButton('💾  Save')
        self.btn_save_lyap.setFixedHeight(28)
        self.btn_save_lyap.setEnabled(False)
        self.btn_save_lyap.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_lyap.clicked.connect(self._save)
        btn_help_lyap = QPushButton('❓  Help')
        btn_help_lyap.setFixedHeight(28)
        btn_help_lyap.setToolTip('FTLE theory & parameters')
        btn_help_lyap.clicked.connect(self._show_help)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_stop_lyap)
        btn_row.addWidget(self.btn_save_lyap)
        btn_row.addWidget(btn_help_lyap)
        lay.addLayout(btn_row)

    # -----------------------------------------------------------------
    def _launch(self):
        """Start FTLE — reuses LyapunovWindow.run() logic directly."""
        self._stop.clear()
        self._running_data = []
        self.prog.setValue(0)
        self.lbl_status.setText('Starting…')
        self.btn_run.setEnabled(False)
        self.btn_save_lyap.setEnabled(False)
        self.lbl_title.setText('FINITE-TIME LYAPUNOV EXPONENT')
        self.lbl_title.setStyleSheet(f'color:{ACCENT};')

        self._T       = self.spn_T.value()
        self._K       = self.spn_K.value()
        self._N_trans = self.spn_tr.value()
        self._eps     = 10 ** self.spn_eps.value()

        # Delegate to LyapunovWindow.run() by temporarily constructing it
        # as a headless object — but simpler: just inline the worker here
        p    = self._p
        hist = self._hist
        T    = self._T; K = self._K; N_trans = self._N_trans; eps = self._eps
        _q   = self._q

        NL       = p['NLattice']
        one_side = p['one_side_fsr']
        D2       = p['D2'];  kin = p['kin'];  kex = p['kex']
        ISite    = p['ISite']; OSite = p['OSite']
        dt       = p['TStep']
        F        = p['F'];   detuning = p['detuning']

        psi_val = p.get('psi', 0.0)
        ls      = p.get('linear_state', {})
        if ls.get('is_cyl') and build_hamiltonian is not None:
            H_full = build_hamiltonian(
                ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2),
                ls.get('phi_iqh1', np.pi/2), ls.get('phi_aqh0', np.pi/4),
                ls.get('phi_aqh1', np.pi/4), psi_val)
            H = H_full[1:, 1:].copy()
        else:
            H = p['H_mat'].copy()
        for k_h, v_h in p.get('heaters', {}).items():
            n = int(k_h.split('_')[1]); H[n-1, n-1] += v_h

        stop_event = self._stop

        def worker():
            try:
                import time
                DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
                    NL, one_side, D2, kin, kex, ISite, OSite)
                props, fc_arr, _ = build_propagators(
                    H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                props_j = jnp.array(props); fc_j = jnp.array(fc_arr)
                stepper = get_stepper(1)

                Psi1 = jnp.array(hist[:, :, -1])
                rng  = np.random.default_rng(42)
                delt = rng.standard_normal(Psi1.shape) + 1j*rng.standard_normal(Psi1.shape)
                delt = delt / np.linalg.norm(delt) * eps
                Psi2 = Psi1 + jnp.array(delt)

                N_total = N_trans + T // K
                log_sum = 0.0; cycle = 0
                running = []

                t0 = time.time()
                for i in range(N_total):
                    if stop_event.is_set():
                        _q.put(('stopped', running)); return

                    for _ in range(K):
                        Psi1 = stepper(Psi1, props_j, fc_j, dt)
                        Psi2 = stepper(Psi2, props_j, fc_j, dt)

                    diff = np.array(Psi2 - Psi1)
                    d    = np.linalg.norm(diff)
                    if d == 0:
                        d = 1e-300

                    Psi2 = Psi1 + jnp.array(eps * diff / d)

                    if i >= N_trans:
                        cycle += 1
                        log_sum += np.log(d / eps)
                        lam = log_sum / (cycle * K)
                        running.append((cycle, lam))
                        elapsed = time.time() - t0
                        _q.put(('progress', cycle, T // K, lam, elapsed))

                _q.put(('done', running))

            except Exception as exc:
                import traceback
                print('LYAPUNOV WORKER ERROR:\n' + traceback.format_exc())
                _q.put(('stopped', []))

        threading.Thread(target=worker, daemon=True).start()
        self._timer.start()

    # -----------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait(); kind = msg[0]
                if kind == 'progress':
                    _, cycle, N, lam, elapsed = msg
                    pct = int(100 * cycle / N)
                    self.prog.setValue(pct)
                    rem = elapsed / cycle * (N - cycle) if cycle > 0 else 0
                    em, es = divmod(int(rem), 60)
                    eh, em2 = divmod(em, 60)
                    eta_str = f'{eh}h{em2:02d}m' if eh else f'{em}m{es:02d}s'
                    self.lbl_status.setText(
                        f'Cycle {cycle}/{N}  |  λ = {lam:.6f}  |  ETA {eta_str}')
                    self._update_plot(lam)
                elif kind in ('done', 'stopped'):
                    _, running = msg
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    if running:
                        lam_final = running[-1][1]
                        self._finish(lam_final, kind == 'stopped')
        except queue.Empty:
            pass

    # -----------------------------------------------------------------
    def _update_plot(self, lam_current):
        self._running_data.append(lam_current)
        cycles = list(range(1, len(self._running_data) + 1))
        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.plot(cycles, self._running_data, color=ACCENT, lw=1.2)
        self.ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        self.ax.set_xlabel('Cycle', color=TEXT_DIM, fontsize=9)
        self.ax.set_ylabel('λ(t)', color=TEXT_DIM, fontsize=9)
        self.ax.set_title(f'Running FTLE  |  λ = {lam_current:.6f}',
                          color=TEXT_COL, fontsize=10, pad=4)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')
        self.canvas.draw_idle()

    # -----------------------------------------------------------------
    def _finish(self, lam_final, stopped):
        self.prog.setValue(100 if not stopped else self.prog.value())
        chaos  = lam_final > 0
        sign   = '> 0  →  CHAOTIC' if chaos else '< 0  →  STABLE'
        color  = '#ff4a6e' if chaos else '#a8ff78'
        self.lbl_title.setText(f'λ = {lam_final:.6f}   ({sign})')
        self.lbl_title.setStyleSheet(f'color:{color};font-size:13px;font-weight:bold;')
        status = 'Stopped' if stopped else 'Done'
        self.lbl_status.setText(
            f'{status}  |  λ_final = {lam_final:.6f}  |  {"Chaotic" if chaos else "Stable"}')
        self.ax.set_title(
            f'FTLE  λ = {lam_final:.6f}  —  {"CHAOTIC" if chaos else "STABLE"}',
            color=color, fontsize=10, pad=4)
        self.canvas.draw_idle()
        self.btn_save_lyap.setEnabled(True)

    # -----------------------------------------------------------------
    def _save(self):
        folder = self._p.get('session_folder')
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)
        lam_final = self._running_data[-1] if self._running_data else None
        chaos     = lam_final is not None and lam_final > 0
        label     = 'chaotic' if chaos else 'stable'
        _flt      = lambda v: f'{v:.4g}'.replace('.', 'p').replace('-', 'm')
        base = f'lyapunov_T{self._T}_K{self._K}_{label}'
        if lam_final is not None:
            base += f'_lam{_flt(lam_final)}'
        for fmt, dpi in [('png', 200), ('svg', 150)]:
            buf = io.BytesIO()
            self.fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            with open(os.path.join(folder, f'{base}.{fmt}'), 'wb') as fh:
                fh.write(buf.getvalue())
        if self._running_data:
            np.savez(os.path.join(folder, f'{base}.npz'),
                     lambda_running = np.array(self._running_data),
                     lambda_final   = np.array([lam_final]),
                     T=np.array([self._T]), K=np.array([self._K]),
                     N_trans=np.array([self._N_trans]),
                     eps=np.array([self._eps]),
                     detuning=np.array([self._p['detuning']]),
                     F=np.array([self._p['F']]),
                     D2=np.array([self._p['D2']]))
        self.lbl_status.setText(f'✅ Saved → {os.path.basename(folder)}')

    # -----------------------------------------------------------------
    def _show_help(self):
        h = QDialog(self); h.setWindowTitle('FTLE — Theory & Parameters')
        h.setWindowFlags(h.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        h.setStyleSheet(STYLE); h.setMinimumSize(640, 700)
        vl = QVBoxLayout(h); vl.setSpacing(8); vl.setContentsMargins(14,14,14,14)
        txt = QPlainTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet('QPlainTextEdit{background:#0a0c14;color:#c8d0e7;'
                          'font-family:"Courier New";font-size:12px;border:1px solid #1e2230;}')
        txt.setPlainText("""\
FINITE-TIME LYAPUNOV EXPONENT (FTLE) ANALYSIS
══════════════════════════════════════════════

WHAT IS IT?
───────────
The FTLE measures the average rate at which two initially nearby
trajectories in phase space diverge (or converge) over a finite
time T. It is the central diagnostic for distinguishing:

  • Stable solutions   →  λ < 0   (perturbations decay)
  • Chaotic solutions  →  λ > 0   (perturbations grow exponentially)
  • Quasi-periodic     →  λ ≈ 0   (marginal, neither)

DERIVATION
──────────
The slow-time LLE evolves the field Ψ ∈ ℂ^(NL×NF) by one map step:

    Ψ(t+1) = F(Ψ(t))

For two nearby trajectories Ψ₁(t) and Ψ₂(t) = Ψ₁(t) + δ(t),
linearising F around Ψ₁ gives:

    δ(t+1) ≈ J(Ψ₁(t)) · δ(t)

where J is the Jacobian of F. After t steps:

    δ(t) = J(t-1) · J(t-2) · … · J(0) · δ(0)

The maximal Lyapunov exponent is:

    λ = lim_{T→∞}  (1/T) · log( ‖δ(T)‖ / ‖δ(0)‖ )

NUMERICAL ALGORITHM (Wolf et al. 1985)
───────────────────────────────────────
Rather than computing J explicitly, we evolve TWO trajectories
and measure their separation directly.

Step 1 — Initialise:
    Ψ₁(0) = last frame of the slow-time history (on attractor)
    δ(0)  = random unit vector × ε     (tiny perturbation)
    Ψ₂(0) = Ψ₁(0) + δ(0)

Step 2 — Transient burn-in (N_trans cycles × K steps):
    Evolve both trajectories and renormalise every K steps,
    but do NOT accumulate log-growth yet. This aligns δ with
    the most unstable direction of the local dynamics.

Step 3 — Accumulate (T/K cycles):
    For each renormalisation cycle i = 1 … T/K:
      a) Evolve Ψ₁ and Ψ₂ forward K slow-time steps.
      b) Measure separation:  dᵢ = ‖Ψ₂ - Ψ₁‖₂
      c) Accumulate:  log_sum += log(dᵢ / ε)
      d) Renormalise:  Ψ₂ ← Ψ₁ + ε · (Ψ₂ - Ψ₁) / dᵢ

Step 4 — Final estimate:
    λ(T) = log_sum / T  (units: per slow-time step)

CONVERGENCE
───────────
The FTLE should converge to the largest real part of the Jacobian
eigenvalues of the continuous-time LLE at the fixed point.
For a soliton: λ_FTLE ≈ λ_max(J) × dt  (near-Goldstone mode).
A clear plateau in the convergence plot indicates convergence.

WHAT λ MEANS FOR EACH SOLUTION TYPE
─────────────────────────────────────
  Stationary soliton:  λ < 0 (clean negative plateau)
  Limit cycle:         λ ≈ 0 (tangential Goldstone mode — ambiguous)
  Chaotic state:       λ > 0 (exponential divergence)

For limit cycles, the Jacobian eigenspectrum is more informative.

PARAMETERS
──────────
T:          Total slow-time steps. Rule of thumb: T ≫ 1/|λ|.
K:          Renormalisation interval.
Transient:  Burn-in cycles before accumulating.
log₁₀(ε):  Perturbation size (ε ≪ field amplitude, e.g. 10⁻¹⁰).
""")
        vl.addWidget(txt)
        btn_close = QPushButton('Close'); btn_close.clicked.connect(h.accept)
        br = QHBoxLayout(); br.addStretch(); br.addWidget(btn_close)
        vl.addLayout(br)
        h.exec_()

    # -----------------------------------------------------------------
    def stop(self):
        self._stop.set()


# =====================================================================
# JACOBIAN TAB  —  continuous-time linearisation eigenspectrum
# =====================================================================
class _JacobianTabWidget(QWidget):
    """Jacobian eigenspectrum analysis tab."""
    def __init__(self, p, hist, parent_win=None):
        super().__init__()
        self._p    = p
        self._hist = hist
        self._stop = threading.Event()
        self._q    = queue.Queue()
        self._timer= QTimer(self); self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)

        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        # ── Parameter row ────────────────────────────────────────────
        param_row = QHBoxLayout()
        def _lbl(t):
            l = QLabel(t); l.setStyleSheet('color:#c8d0e7;font-size:12px;'); return l

        self.spn_step = QSpinBox()
        self.spn_step.setRange(0, max(0, hist.shape[2]-1))
        self.spn_step.setValue(hist.shape[2]-1)
        self.spn_step.setToolTip('Slow-time step to use as fixed-point seed')

        param_row.addWidget(_lbl('Seed step:')); param_row.addWidget(self.spn_step)

        # Show the operating point — updates live as step changes
        def _sched_at(val, key, fallback):
            ch = p.get('schedule', {})
            arr = ch.get(key)
            if arr is not None and val < len(arr):
                return float(arr[val])
            return float(p.get(fallback, float('nan')))

        init_step = self.spn_step.value()
        self.lbl_op = QLabel(
            f'Δ = {_sched_at(init_step,"detuning","detuning"):.4f}   '
            f'F = {_sched_at(init_step,"F","F"):.4f}')
        self.lbl_op.setStyleSheet('color:#ff8c42;font-size:12px;font-style:italic;')
        self.lbl_op.setToolTip('Δ and F at the chosen step (from schedule)')
        param_row.addSpacing(16)
        param_row.addWidget(self.lbl_op)

        def _on_step_changed(val):
            self.lbl_op.setText(
                f'Δ = {_sched_at(val,"detuning","detuning"):.4f}   '
                f'F = {_sched_at(val,"F","F"):.4f}')
        self.spn_step.valueChanged.connect(_on_step_changed)
        param_row.addStretch()

        # LM fixed-point solve controls
        from PyQt5.QtWidgets import QCheckBox
        self.chk_lm = QCheckBox('Use LM')
        self.chk_lm.setChecked(False)
        self.chk_lm.setToolTip('Run Levenberg-Marquardt fixed-point solver before Jacobian.\nOnly meaningful for steady states — disable for chaotic/breather fields.')
        self.chk_lm.setStyleSheet('color:#c8d0e7;font-size:12px;')
        param_row.addWidget(self.chk_lm)

        param_row.addWidget(_lbl('LM tol (log₁₀):'))
        self.spn_tol = QSpinBox()
        self.spn_tol.setRange(-14, -3); self.spn_tol.setValue(-10)
        self.spn_tol.setToolTip('Levenberg-Marquardt convergence tolerance: ‖f‖ < 10^tol')
        self.spn_tol.setFixedWidth(55)
        param_row.addWidget(self.spn_tol)

        param_row.addWidget(_lbl('max iter:'))
        self.spn_lm_iter = QSpinBox()
        self.spn_lm_iter.setRange(1, 500); self.spn_lm_iter.setValue(50)
        self.spn_lm_iter.setToolTip('Maximum LM iterations to find fixed point')
        self.spn_lm_iter.setFixedWidth(55)
        param_row.addWidget(self.spn_lm_iter)

        param_row.addSpacing(8)
        self.btn_run = QPushButton('▶  Run Jacobian')
        self.btn_run.setFixedHeight(26)
        self.btn_run.setStyleSheet(
            'QPushButton:enabled{color:#a8ff78;border-color:#a8ff78;}'
            'QPushButton:disabled{color:#4a5270;border-color:#2a3050;background:#141a28;}')
        self.btn_run.clicked.connect(self._launch)
        param_row.addWidget(self.btn_run)

        self.btn_stop_jac = QPushButton('■  Stop')
        self.btn_stop_jac.setFixedHeight(26)
        self.btn_stop_jac.setEnabled(False)
        self.btn_stop_jac.clicked.connect(lambda: self._stop.set())
        param_row.addWidget(self.btn_stop_jac)

        lay.addLayout(param_row)

        # ── Status label + indeterminate progress bar ─────────────────
        self.lbl_status = QLabel('Ready.')
        self.lbl_status.setStyleSheet('color:#4a5270;font-size:12px;')
        lay.addWidget(self.lbl_status)

        self.prog_jac = QProgressBar()
        self.prog_jac.setRange(0, 0)      # 0,0 = indeterminate/marquee
        self.prog_jac.setFixedHeight(4)
        self.prog_jac.setTextVisible(False)
        self.prog_jac.setStyleSheet(
            'QProgressBar{background:#0d0f1a;border:none;border-radius:2px;}'
            'QProgressBar::chunk{background:#00e5ff;border-radius:2px;}')
        self.prog_jac.setVisible(False)   # hidden until running
        lay.addWidget(self.prog_jac)

        # ── 2×2 figure: top row = eigenspectrum (full + zoom),
        #              bottom row = LM-polished field (|a_W|² dB + |a_T|²)
        self.fig = Figure(facecolor=DARK_BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.08,
                                 wspace=0.32, hspace=0.50)
        gs_jac = self.fig.add_gridspec(2, 2)
        self.ax_full  = self.fig.add_subplot(gs_jac[0, 0])
        self.ax_zoom  = self.fig.add_subplot(gs_jac[0, 1])
        self.ax_field_W = self.fig.add_subplot(gs_jac[1, 0])   # |a_W|² dB
        self.ax_field_T = self.fig.add_subplot(gs_jac[1, 1])   # |a_T|² linear
        for ax in (self.ax_full, self.ax_zoom):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.axvline(0, color='white', lw=0.6, ls='--', alpha=0.4)
            ax.axhline(0, color='white', lw=0.4, ls='-', alpha=0.2)
            ax.set_xlabel('Re(λ)', color=TEXT_DIM, fontsize=9)
            ax.set_ylabel('Im(λ)', color=TEXT_DIM, fontsize=9)
        self.ax_full.set_title('full eigenspectrum', color=TEXT_COL, fontsize=9, pad=3)
        self.ax_zoom.set_title('zoom: near Re = 0', color=TEXT_COL, fontsize=9, pad=3)
        # Bottom row styling (matches the NonlinearWindow cross-section panels)
        for ax in (self.ax_field_W, self.ax_field_T):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_field_W.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=9)
        self.ax_field_W.set_title('10·log|a_W|²  (LM-polished, output ring)',
                                  color='#ff8c42', fontsize=9, pad=3)
        self.ax_field_T.set_xlabel('θ index', color=TEXT_DIM, fontsize=9)
        self.ax_field_T.set_title('|a_T|²  (LM-polished, output ring)',
                                  color='#a8ff78', fontsize=9, pad=3)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, stretch=1)

        # Navigation toolbar: interactive pan, zoom-to-rectangle, home, save.
        # Lets you drag/zoom into the near-0 region on the spectrum plots.
        # Home button restores the auto-zoom set by _plot().
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar_jac = NavigationToolbar2QT(self.canvas, self)
        self.toolbar_jac.setStyleSheet(
            'QToolBar{background:#141a28;border:1px solid #3a4560;spacing:2px;}'
            'QToolButton{background:#1e2538;color:#c8d0e7;border:1px solid #3a4560;'
            '  border-radius:2px;padding:3px;}'
            'QToolButton:hover{background:#2a3148;border-color:#00e5ff;}'
            'QToolButton:checked{background:#00e5ff;color:#0e1520;border-color:#00e5ff;}'
            'QLabel{color:#c8d0e7;font-size:11px;background:#141a28;}')
        lay.addWidget(self.toolbar_jac)

        # ── Result label + Save button ────────────────────────────────
        self.lbl_result = QLabel('')
        self.lbl_result.setFont(QFont('Courier New', 11, QFont.Bold))
        lay.addWidget(self.lbl_result)

        bot_row = QHBoxLayout()
        self.btn_save_jac = QPushButton('💾  Save eigenvalues')
        self.btn_save_jac.setFixedHeight(28)
        self.btn_save_jac.setEnabled(False)
        self.btn_save_jac.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_jac.clicked.connect(self._save)
        btn_help_jac = QPushButton('❓  Help')
        btn_help_jac.setFixedHeight(28)
        btn_help_jac.setToolTip('Jacobian eigenspectrum — theory & interpretation')
        btn_help_jac.clicked.connect(self._show_help)
        bot_row.addStretch()
        bot_row.addWidget(self.btn_save_jac)
        bot_row.addWidget(btn_help_jac)
        lay.addLayout(bot_row)

        # Storage for last computed eigenvalues + metadata
        self._last_ev    = None
        self._last_field = None     # LM-polished field (NL, NF) in μ-basis
        self._last_meta  = {}

    # -----------------------------------------------------------------
    def _launch(self):
        self._stop.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop_jac.setEnabled(True)
        self.lbl_status.setText('Building Jacobian…')
        self.lbl_result.setText('')
        self.prog_jac.setVisible(True)
        step     = self.spn_step.value()
        tol_val  = 10.0 ** self.spn_tol.value()
        iter_val = self.spn_lm_iter.value()
        use_lm   = self.chk_lm.isChecked()

        p    = self._p
        hist = self._hist
        q    = self._q

        def worker():
            try:
                NL       = p['NLattice']
                one_side = p['one_side_fsr']
                D2       = p['D2']
                kin      = p['kin'];  kex = p['kex']
                ISite    = p['ISite']
                NF       = 2*one_side + 1
                NLNF     = NL * NF

                # Read all swept parameters at the chosen step from schedule.
                # Falls back to launch-step values if schedule not present.
                sched_ch = p.get('schedule', {})

                def _at(key, fallback):
                    arr = sched_ch.get(key)
                    if arr is not None and step < len(arr):
                        return float(arr[step])
                    return float(fallback)

                Delta   = _at('detuning', p['detuning'])
                F_val   = _at('F',        p['F'])
                psi_val = _at('psi',      p.get('psi', 0.0))

                # Heaters at the chosen step
                heaters_at_step = {}
                for k, arr in sched_ch.items():
                    if k.startswith('heater_'):
                        heaters_at_step[k] = float(arr[step]) if step < len(arr) else 0.0
                if not heaters_at_step:
                    heaters_at_step = p.get('heaters', {})

                ls = p.get('linear_state', {})
                if ls.get('is_cyl') and build_hamiltonian is not None:
                    H_full = build_hamiltonian(
                        ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                        ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2),
                        ls.get('phi_iqh1', np.pi/2), ls.get('phi_aqh0', np.pi/4),
                        ls.get('phi_aqh1', np.pi/4), psi_val)
                    H = H_full[1:, 1:].copy()
                else:
                    H = p['H_mat'].copy()
                for k_h, v_h in heaters_at_step.items():
                    n = int(k_h.split('_')[1]); H[n-1, n-1] += v_h
                Hr = H.real; Hi = H.imag

                LM = np.full((NL, NF), kin, dtype=float)
                LM[ISite-1, :] += kex
                OSite = p.get('OSite', ISite)
                if OSite != ISite: LM[OSite-1, :] += kex

                # μ in FFT order: [0, 1, ..., +N, -N, ..., -1]
                mu_fft   = np.round(np.fft.fftfreq(NF, d=1.0) * NF).astype(float)
                # Dispersion in μ-basis is DIAGONAL with entries (D₂/2)·μ².
                # (A circulant from ifft(μ²) would be the θ-basis kernel for
                #  ∂²/∂θ² — wrong to apply to μ-domain state vectors.)
                Dsp      = np.diag((D2/2.0) * mu_fft**2)
                # Pump forcing must match what the split-step stepper applies.
                # The stepper writes `fv[ISite-1] = F` at array-index PumpFSR in
                # the μ-domain — i.e. it treats F as already the μ-space amplitude.
                # Therefore the Jacobian-tab residual must use the same convention:
                # pump_A[ISite-1, 0] = F (no √NF rescaling).
                PumpFSR  = 0
                pump_A   = np.zeros((NL, NF))
                pump_A[ISite-1, PumpFSR] = F_val

                I_NF     = np.eye(NF); I_NL = np.eye(NL)
                Hi_kron  = np.kron(Hi, I_NF)
                Hr_kron  = np.kron(Hr, I_NF)
                Dsp_kron = np.kron(I_NL, Dsp)

                tol      = tol_val
                max_iter = iter_val

                # Pre-compute flat arrays needed by residual_and_jac
                Lf     = LM.ravel()
                pump_re_flat = pump_A.ravel()
                pump_im_flat = np.zeros(NLNF)

                # ── Levenberg-Marquardt fixed-point solve ─────────────────
                # True nonlinear residual f(a) = da/dt = 0 at steady state,
                # AND the correct Jacobian of that residual.
                #
                # EOM (FSR domain, continuous):
                #   da_{m,mu}/dt = (iΔ - iD_mu - L_m)*a_{m,mu}
                #                  - i*sum_m' H_{mm'}*a_{m',mu}
                #                  + i*[FFT(|a_T|^2 * a_T)]_{m,mu}    (Kerr in θ)
                #                  + F_{m,mu}
                #
                # State u = (A_W, B_W)  (real/imag parts in FSR domain).
                #
                # KERR JACOBIAN — the bug fix:
                # The Kerr term is local in θ, NOT in μ. Its linearization in
                # the μ-domain state is therefore DENSE in μ (block-diagonal
                # in site n, dense N_F×N_F per site). The previous code used
                # diag(2*A_W*B_W) etc., which would only be correct for a
                # μ-local SPM — i.e. it silently dropped all four-wave-mixing
                # coupling between FSRs and missed modulational instability.
                #
                # Derivation (per site n, with a_T = ifft(a_W, norm='ortho')):
                #   ∂(|a_T|² a_T)/∂a_T   = 2|a_T|²
                #   ∂(|a_T|² a_T)/∂a_T*  = a_T²
                # so for the FSR-domain Kerr  N_W = i·F{|a_T|² a_T}:
                #   ∂N_W/∂a_W   = i · F · diag(2|a_T|²) · F⁻¹  ≡ i·P_n
                #   ∂N_W/∂a_W*  = i · F · diag( a_T² ) · F    ≡ i·Q_n
                # (note: Q has F on both sides, NOT F⁻¹ on the right —
                #  comes from ∂a_T*/∂a_W* = (F⁻¹)* = F for unitary F.)
                # where F here is the orthonormal DFT matrix.
                # Convert to (A,B) blocks via ∂_A = ∂_a + ∂_a*, ∂_B = i(∂_a - ∂_a*):
                #   K11 = -Im(P+Q)   K12 = -Re(P-Q)
                #   K21 = +Re(P+Q)   K22 = -Im(P-Q)

                # Pre-compute the orthonormal DFT matrix once (NF×NF).
                # F[k,j] = exp(-2πi k j / NF) / √NF   (matches np.fft.fft norm='ortho').
                _kk, _jj = np.meshgrid(np.arange(NF), np.arange(NF), indexing='ij')
                Fmat  = np.exp(-2j*np.pi*_kk*_jj / NF) / np.sqrt(NF)   # (NF,NF)
                Fmati = Fmat.conj().T                                   # F⁻¹ (unitary)

                def _kerr_blocks(a_T):
                    """Build the dense per-site Kerr Jacobian blocks K11,K12,K21,K22.
                    Each is shape (NL*NF, NL*NF), block-diagonal in site n."""
                    K11 = np.zeros((NLNF, NLNF))
                    K12 = np.zeros((NLNF, NLNF))
                    K21 = np.zeros((NLNF, NLNF))
                    K22 = np.zeros((NLNF, NLNF))
                    for n in range(NL):
                        an   = a_T[n, :]                       # (NF,)
                        Pn   = (Fmat * (2.0*np.abs(an)**2)) @ Fmati   # F·diag(2|a|²)·F⁻¹
                        Qn   = (Fmat * (an**2))               @ Fmat    # F·diag(a²)·F  (NOT F⁻¹)
                        s    = slice(n*NF, (n+1)*NF)
                        K11[s, s] = -(Pn + Qn).imag
                        K12[s, s] = -(Pn - Qn).real
                        K21[s, s] = +(Pn + Qn).real
                        K22[s, s] = -(Pn - Qn).imag
                    return K11, K12, K21, K22

                # Pre-compute the linear (Kerr-independent) part of the Jacobian.
                # It does NOT depend on the field, so we build it once.
                alph    = Delta * np.eye(NLNF)
                Lf_diag = np.diag(Lf)
                J11_lin = Hi_kron - Lf_diag
                J12_lin = -alph + Dsp_kron + Hr_kron
                J21_lin =  alph - Dsp_kron - Hr_kron
                J22_lin = Hi_kron - Lf_diag

                def residual_and_jac(A_flat, B_flat):
                    """True nonlinear residual da/dt and its (correct) Jacobian."""
                    A_W = A_flat.reshape(NL, NF)
                    B_W = B_flat.reshape(NL, NF)
                    a_W = A_W + 1j*B_W

                    # FSR → θ for the Kerr nonlinearity (local in θ).
                    a_T   = np.fft.ifft(a_W, axis=1, norm='ortho')
                    pwr_T = np.abs(a_T)**2
                    kerr_T = pwr_T * a_T
                    kerr_W = np.fft.fft(kerr_T, axis=1, norm='ortho')

                    # Linear part in FSR domain (matrix-vector form).
                    lin_re = (J11_lin @ A_flat) + (J12_lin @ B_flat)
                    lin_im = (J21_lin @ A_flat) + (J22_lin @ B_flat)

                    # Kerr contribution: +i·kerr_W → Re = -Im(kerr_W), Im = +Re(kerr_W)
                    kerr_re = -kerr_W.imag.ravel()
                    kerr_im = +kerr_W.real.ravel()

                    # Pump (delta at μ_p, real)
                    pump_re = pump_A.ravel()

                    f1 = lin_re + kerr_re + pump_re
                    f2 = lin_im + kerr_im
                    f_ = np.concatenate([f1, f2])

                    # Jacobian: linear part + dense Kerr blocks (block-diagonal in site).
                    K11, K12, K21, K22 = _kerr_blocks(a_T)
                    Jac_ = np.block([[J11_lin + K11, J12_lin + K12],
                                     [J21_lin + K21, J22_lin + K22]])
                    return f_, Jac_

                # Field from simulation snapshot → FSR domain
                a_field = hist[:, :, step]                           # (NL,NF) theta domain
                a_field = np.fft.fft(a_field, axis=1, norm='ortho') # → FSR domain
                A_flat  = a_field.real.ravel().copy()
                B_flat  = a_field.imag.ravel().copy()

                if use_lm:
                    lam_lm = 1.0
                    lm_converged = False
                    lm_iters = 0
                    f0, _ = residual_and_jac(A_flat, B_flat)

                    for lm_iters in range(1, max_iter + 1):
                        if self._stop.is_set():
                            q.put(('stopped',)); return

                        f_cur, Jac_cur = residual_and_jac(A_flat, B_flat)
                        res_cur = np.linalg.norm(f_cur)

                        q.put(('status',
                               f'LM iter {lm_iters}/{max_iter}  '
                               f'‖f‖={res_cur:.3e}  λ={lam_lm:.2e}'))

                        if res_cur < tol:
                            lm_converged = True
                            break

                        JtJ  = Jac_cur.T @ Jac_cur
                        Jtf  = Jac_cur.T @ f_cur
                        try:
                            delta = np.linalg.solve(JtJ + lam_lm * np.eye(2*NLNF), -Jtf)
                        except np.linalg.LinAlgError:
                            lam_lm *= 10.0
                            continue

                        u_new = np.concatenate([A_flat, B_flat]) + delta
                        A_new = u_new[:NLNF]; B_new = u_new[NLNF:]
                        f_new, _ = residual_and_jac(A_new, B_new)
                        res_new  = np.linalg.norm(f_new)

                        if res_new < res_cur:
                            A_flat = A_new; B_flat = B_new
                            lam_lm = max(lam_lm / 3.0, 1e-12)
                        else:
                            lam_lm = min(lam_lm * 10.0, 1e12)

                    res_final = np.linalg.norm(residual_and_jac(A_flat, B_flat)[0])
                    if lm_converged:
                        q.put(('status',
                               f'✅ LM converged in {lm_iters} iters  '
                               f'‖f‖={res_final:.3e}  (tol={tol:.0e})  '
                               f'— building Jacobian…'))
                    else:
                        q.put(('status',
                               f'⚠️  LM did not converge ({max_iter} iters)  '
                               f'‖f‖={res_final:.3e} > {tol:.0e}  '
                               f'— Jacobian at best available point…'))
                else:
                    q.put(('status',
                           f'Δ={Delta:.4f}  F={F_val:.4f}  '
                           f'— using snapshot directly (LM disabled)…'))

                # Build Jacobian at current (snapshot or LM-converged) point
                a_field = (A_flat + 1j * B_flat).reshape(NL, NF)

                if self._stop.is_set():
                    q.put(('stopped',)); return

                q.put(('status',
                       f'Δ={Delta:.4f}  F={F_val:.4f}  —  '
                       f'building {2*NLNF}×{2*NLNF} Jacobian at step {step}…'))

                # Reuse residual_and_jac to guarantee the Jacobian here matches
                # exactly the one used inside LM — single source of truth.
                _, J = residual_and_jac(A_flat, B_flat)

                if self._stop.is_set():
                    q.put(('stopped',)); return

                # ── Estimate duration based on workstation calibration ────
                # Empirical scaling: np.linalg.eig is O(N³). For your Dell
                # Precision 5860 (Xeon w5-2565X, 14 cores, DDR5-4800, MKL),
                # a 2050×2050 complex dense eig takes roughly 2.3 seconds.
                # Scaling: t(N) ≈ 2.3 · (N/2050)³
                import time as _time
                N_j         = int(J.shape[0])
                _est_sec    = 2.3 * (N_j / 2050.0) ** 3
                _start_ts   = _time.time()
                _start_wall = _time.strftime('%H:%M:%S',
                                             _time.localtime(_start_ts))
                def _fmt(s):
                    s = max(int(s), 0)
                    h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
                    if h: return f'{h}h {m}m {s}s'
                    if m: return f'{m}m {s}s'
                    return f'{s}s'
                _eta_str = _fmt(_est_sec)
                _end_est = _time.strftime('%H:%M:%S',
                    _time.localtime(_start_ts + _est_sec))

                q.put(('status',
                       f'Δ={Delta:.4f}  F={F_val:.4f}  —  '
                       f'Jacobian built ({N_j}×{N_j})  —  '
                       f'started {_start_wall}, est ~{_eta_str} '
                       f'(≈ done {_end_est})'))

                ev = np.linalg.eigvals(J)

                # Capture actual timing for display in _plot
                _end_ts       = _time.time()
                _actual_sec   = _end_ts - _start_ts
                _end_wall     = _time.strftime('%H:%M:%S',
                                               _time.localtime(_end_ts))
                _timing_info  = dict(
                    start_wall  = _start_wall,
                    end_wall    = _end_wall,
                    actual_sec  = _actual_sec,
                    actual_str  = _fmt(_actual_sec),
                    est_sec     = _est_sec,
                    est_str     = _eta_str,
                )
                # Pass back both eigenvalues AND the LM-polished field (μ-basis)
                # so the UI can display what LM converged to.
                q.put(('done', ev, 0.0, a_field.copy(), _timing_info))

            except Exception as exc:
                import traceback; traceback.print_exc()
                q.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self._timer.start()

    # -----------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == 'status':
                    self.lbl_status.setText(msg[1])
                elif kind == 'done':
                    # Unpack with backward compat: old 4-tuple still works.
                    if len(msg) >= 5:
                        _, ev, rhs_norm, a_field_polished, timing_info = msg
                    else:
                        _, ev, rhs_norm, a_field_polished = msg
                        timing_info = None
                    self._last_timing  = timing_info
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_jac.setEnabled(False)
                    self._last_ev      = ev
                    self._last_field   = a_field_polished   # μ-basis, (NL, NF)
                    self._last_meta = dict(
                        step    = self.spn_step.value(),
                        Delta   = float(self._p.get('detuning', float('nan'))),
                        F       = float(self._p.get('F',        float('nan'))),
                        D2      = float(self._p.get('D2',       float('nan'))),
                        kin     = float(self._p.get('kin',      float('nan'))),
                        kex     = float(self._p.get('kex',      float('nan'))),
                        NLattice= int(self._p.get('NLattice',   1)),
                        one_side= int(self._p.get('one_side_fsr', 128)),
                    )
                    # Override Delta/F from schedule if available
                    sched_ch = self._p.get('schedule', {})
                    step = self._last_meta['step']
                    for key, meta_key in [('detuning','Delta'),('F','F')]:
                        arr = sched_ch.get(key)
                        if arr is not None and step < len(arr):
                            self._last_meta[meta_key] = float(arr[step])
                    self.btn_save_jac.setEnabled(True)
                    self.prog_jac.setVisible(False)
                    self._plot(ev, rhs_norm)
                elif kind == 'stopped':
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_jac.setEnabled(False)
                    self.prog_jac.setVisible(False)
                    self.lbl_status.setText('Stopped.')
                elif kind == 'error':
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_jac.setEnabled(False)
                    self.prog_jac.setVisible(False)
                    self.lbl_status.setText(f'Error: {msg[1]}')
        except queue.Empty:
            pass

    # -----------------------------------------------------------------
    def _plot(self, ev, rhs_norm):
        re = ev.real; im = ev.imag
        n_unstable = int(np.sum(re > 1e-8))
        max_re = float(re.max()); min_re = float(re.min())
        verdict = 'STABLE' if n_unstable == 0 else 'UNSTABLE'
        color   = '#a8ff78' if n_unstable == 0 else '#ff4a6e'

        # Classify
        re_sorted = np.sort(np.unique(np.round(re, 5)))
        bulk_re   = float(re_sorted[len(re_sorted)//2])   # most common = middle unique val

        def pt_color(r):
            if abs(r - max_re) < abs(max_re)*0.01 + 1e-7 and max_re != bulk_re:
                return '#a8ff78'    # near-Goldstone (least negative)
            if abs(r - min_re) < abs(min_re)*0.01 + 1e-7 and min_re != bulk_re:
                return '#ff8c42'    # most-damped outlier
            if r > 1e-8:
                return '#ff4a6e'    # unstable
            return '#378ADD'        # bulk

        colors = [pt_color(r) for r in re]

        kin_val = float(self._p.get('kin', float('nan')))
        kex_val = float(self._p.get('kex', float('nan')))
        k_tot   = kin_val + kex_val

        # κ_σ per supermode of H.
        # Prefer a fresh recompute from H_mat because the linear_state cache may
        # be stale if it was saved with an older (buggy) version of Linear.py.
        # Fall back to the cached value only if we can't recompute.
        kappa_sigma = None
        H_mat = self._p.get('H_mat')
        ISite = self._p.get('ISite', 1)
        OSite = self._p.get('OSite', 1)
        if H_mat is not None and not np.isnan(kin_val) and not np.isnan(kex_val):
            try:
                _, V_H = np.linalg.eigh(H_mat)
                # κ_σ = κ_in + Σ_{s ∈ port_sites} κ_ex · |⟨σ|s⟩|²
                # Use unique port sites so ISite==OSite isn't double-counted.
                kappa_sigma = np.full(V_H.shape[1], kin_val, dtype=float)
                for port in {ISite, OSite}:
                    kappa_sigma += kex_val * np.abs(V_H[port-1, :])**2
            except Exception:
                pass

        if kappa_sigma is None:
            # Last-ditch fallback: use the linear_state cache
            ls_state = self._p.get('linear_state', {})
            sp_state = ls_state.get('spectrum', {}) if ls_state else {}
            if sp_state:
                kappa_sigma = sp_state.get('kappa_sigma')

        for ax in (self.ax_full, self.ax_zoom):
            ax.cla()
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.axvline(0, color='white', lw=0.6, ls='--', alpha=0.4)
            ax.axhline(0, color='white', lw=0.4, alpha=0.2)
            if kappa_sigma is not None:
                for k_s in kappa_sigma:
                    ax.axvline(-k_s, color='#ff8c42', lw=1.0, ls=':', alpha=0.8)
            ax.set_xlabel('Re(λ)', color=TEXT_DIM, fontsize=9)
            ax.set_ylabel('Im(λ)', color=TEXT_DIM, fontsize=9)

        self.ax_full.scatter(re, im, c=colors, s=8, linewidths=0)
        self.ax_full.set_title(
            f'full eigenspectrum  ({len(ev)} eigs)  —  {verdict}',
            color=color, fontsize=9, pad=3)
        self.ax_full.legend(fontsize=7, frameon=False, labelcolor=TEXT_COL,
                            loc='upper left')

        # ── Zoom panel: center on Re(λ)=0, adaptive to actual mode positions ──
        # Goal: show the near-Goldstone region clearly. If the grid is well-
        # resolved, the Goldstone sits within ~10⁻⁶ of the origin and this zoom
        # reveals it. If the grid is under-resolved, the nearest-to-zero modes
        # sit at Re ≈ -κ and the zoom shows them with their imaginary splitting.
        #
        # Gap-aware scaling: sort distances from origin in log-space and find
        # the biggest jump (in log) between consecutive eigenvalues. The zoom
        # window is set just past the last mode before that jump. This naturally
        # separates a near-zero cluster from the bulk, no matter where the
        # bulk is located. If no gap exists (eigenvalues span continuously from
        # 0 to κ), fall back to a sensible fixed window of 2×κ.
        k_show       = 20                            # how many near-zero modes to emphasize
        dist_to_zero = np.abs(ev)
        sorted_d     = np.sort(dist_to_zero)
        # Log-space gap detection on the smallest 40 distances
        n_probe = min(40, len(sorted_d))
        log_d   = np.log10(np.maximum(sorted_d[:n_probe], 1e-16))
        gaps    = np.diff(log_d)
        if len(gaps) > 0 and gaps.max() > 1.5:       # significant gap = factor of ~30×
            i_gap      = int(np.argmax(gaps))
            scale_hint = sorted_d[i_gap]              # last mode before the gap
        else:
            # No clear near-zero cluster; use 1/3 of κ_tot as default
            scale_hint = max(k_tot / 3.0, 1e-6)
        half_w_re = max(3.0 * scale_hint, 1e-8)
        half_w_re = min(half_w_re, 2 * k_tot)         # never zoom out past 2κ on Re
        half_w_im = half_w_re * 4                      # allow elongated Im range
        half_w_im = min(max(half_w_im, 3 * scale_hint, 1e-6), 5.0)

        # Mask to zoom window
        mask = (re > -half_w_re) & (re < +half_w_re) & \
               (np.abs(im) < half_w_im)
        self.ax_zoom.scatter(re[mask], im[mask],
                             c=[colors[i] for i in range(len(colors)) if mask[i]],
                             s=25, linewidths=0, zorder=2)
        # Highlight the k_show eigenvalues nearest the origin with a bright ring
        idx_near = np.argsort(dist_to_zero)[:k_show]
        self.ax_zoom.scatter(re[idx_near], im[idx_near],
                             facecolors='none', edgecolors='#ffd700',
                             s=80, linewidths=1.2, zorder=4,
                             label=f'{k_show} nearest to 0')
        # Detect visually-overlapping eigenvalues and annotate multiplicity.
        # Two eigenvalues "overlap" if their separation in either axis is below
        # 1% of the zoom window half-width (within a few pixels at typical DPI).
        re_tol = 0.01 * half_w_re
        im_tol = 0.01 * half_w_im
        visited = np.zeros(len(idx_near), dtype=bool)
        for j, i in enumerate(idx_near):
            if visited[j]: continue
            # Find all indices in idx_near that are within tolerance of ev[i]
            cluster = [k for k, ki in enumerate(idx_near)
                       if (abs(re[ki] - re[i]) < re_tol and
                           abs(im[ki] - im[i]) < im_tol)]
            for k in cluster: visited[k] = True
            if len(cluster) > 1:
                # Annotate "×N" next to the cluster center
                cx = np.mean([re[idx_near[k]] for k in cluster])
                cy = np.mean([im[idx_near[k]] for k in cluster])
                # Offset label to the upper-right of the cluster
                self.ax_zoom.annotate(f'×{len(cluster)}',
                                      xy=(cx, cy),
                                      xytext=(6, 6), textcoords='offset points',
                                      color='#ffd700', fontsize=9,
                                      fontweight='bold', zorder=5)
        # Mark the origin with a bright crosshair
        self.ax_zoom.axvline(0, color='#ffd700', lw=0.8, alpha=0.5, zorder=3)
        self.ax_zoom.axhline(0, color='#ffd700', lw=0.8, alpha=0.5, zorder=3)

        self.ax_zoom.set_xlim(-half_w_re, +half_w_re)
        self.ax_zoom.set_ylim(-half_w_im, +half_w_im)
        self.ax_zoom.legend(fontsize=7, frameon=False, labelcolor=TEXT_COL,
                            loc='upper left')
        self.ax_zoom.set_title(
            f'zoom: Re ∈ ±{half_w_re:.1e}, Im ∈ ±{half_w_im:.1e}',
            color=TEXT_COL, fontsize=9, pad=3)

        # ── Bottom row: LM-polished field  |a_W|² (dB)  and  |a_T|² (linear) ──
        self._draw_polished_field()

        self.canvas.draw_idle()

        # Build status line: Re(λ) range + loss info + (optionally) timing
        _status = (f'Re(λ) ∈ [{min_re:+.5f}, {max_re:+.5f}]  |  '
                   f'{n_unstable} unstable  |  '
                   f'kin={kin_val:.4f}  kex={kex_val:.4f}')
        _ti = getattr(self, '_last_timing', None)
        if _ti:
            _status += (f'  |  eig: started {_ti["start_wall"]}, '
                        f'ended {_ti["end_wall"]}, '
                        f'took {_ti["actual_str"]} '
                        f'(est {_ti["est_str"]})')
        self.lbl_status.setText(_status)
        self.lbl_result.setText(f'λ_max = {max_re:+.6e}   →   {verdict}')
        self.lbl_result.setStyleSheet(f'color:{color};font-size:12px;font-weight:bold;')

    # -----------------------------------------------------------------
    def _draw_polished_field(self):
        """Draw the LM-polished field on the bottom row:
           left = |a_W|² in dB (μ-axis fftshifted), right = |a_T|² linear (θ-axis)."""
        # Clear and re-style both axes every call so replots are clean
        for ax, title, color_ in [
            (self.ax_field_W, '10·log|a_W|²  (LM-polished, output ring)', '#ff8c42'),
            (self.ax_field_T, '|a_T|²  (LM-polished, output ring)',        '#a8ff78'),
        ]:
            ax.cla()
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.set_title(title, color=color_, fontsize=9, pad=3)

        a_field = getattr(self, '_last_field', None)
        if a_field is None:
            self.ax_field_W.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=9)
            self.ax_field_T.set_xlabel('θ index', color=TEXT_DIM, fontsize=9)
            return   # no field yet (pre-run state)

        # Pick the output-ring row — matches the cross-section panel convention
        OSite = int(self._p.get('OSite', 1))
        NL, NF = a_field.shape
        row_idx = max(0, min(OSite - 1, NL - 1))
        a_W_row = a_field[row_idx]            # (NF,) μ-basis, fft ordering
        # Convert to θ via inverse FFT (orthonormal convention, consistent with
        # how the rest of the codebase displays intra-cavity intensity).
        a_T_row = np.fft.ifft(a_W_row, norm='ortho')

        # ── |a_W|² in dB ─────────────────────────────────────────────
        # Shift to monotonic μ for a clean symmetric axis [-N, 0, +N]
        pW_dB = 10.0 * np.log10(np.abs(a_W_row)**2 + 1e-20)
        mu_axis = np.fft.fftfreq(NF) * NF           # ...-N/2 ... 0 ... +N/2 ...
        mu_axis = np.round(mu_axis).astype(int)
        order = np.argsort(mu_axis)
        self.ax_field_W.plot(mu_axis[order], pW_dB[order],
                             color='#ff8c42', lw=1.1)
        one_side = NF // 2
        self.ax_field_W.set_xlim(-one_side, one_side)
        self.ax_field_W.set_xticks([-one_side, 0, one_side])
        self.ax_field_W.set_xticklabels([f'-{one_side}', '0', f'+{one_side}'])
        self.ax_field_W.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=9)

        # ── |a_T|² linear ────────────────────────────────────────────
        pT = np.abs(a_T_row)**2
        self.ax_field_T.plot(np.arange(NF), pT, color='#a8ff78', lw=1.1)
        self.ax_field_T.set_xlim(0, NF - 1)
        self.ax_field_T.set_xticks([0, NF - 1])
        self.ax_field_T.set_xticklabels(['0', '2π'])
        self.ax_field_T.set_xlabel('θ index', color=TEXT_DIM, fontsize=9)

    # -----------------------------------------------------------------
    def _save(self):
        if self._last_ev is None:
            return
        folder = self._p.get('session_folder')
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)

        m    = self._last_meta
        step = m['step']
        _flt = lambda v: f'{v:.4g}'.replace('.', 'p').replace('-', 'm')
        base = (f'jacobian_step{step}'
                f'_D{_flt(m["Delta"])}'
                f'_F{_flt(m["F"])}')

        # ── PNG + SVG ─────────────────────────────────────────────────
        for fmt, dpi in [('png', 200), ('svg', 150)]:
            buf = io.BytesIO()
            self.fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            with open(os.path.join(folder, f'{base}.{fmt}'), 'wb') as fh:
                fh.write(buf.getvalue())

        # ── NPZ ───────────────────────────────────────────────────────
        ev   = self._last_ev
        path = os.path.join(folder, f'{base}.npz')
        np.savez(path,
                 eigenvalues_re = ev.real,
                 eigenvalues_im = ev.imag,
                 step           = np.array([step]),
                 Delta          = np.array([m['Delta']]),
                 F              = np.array([m['F']]),
                 D2             = np.array([m['D2']]),
                 kin            = np.array([m['kin']]),
                 kex            = np.array([m['kex']]),
                 NLattice       = np.array([m['NLattice']]),
                 one_side_fsr   = np.array([m['one_side']]),
                 max_re         = np.array([float(ev.real.max())]),
                 min_re         = np.array([float(ev.real.min())]),
                 n_unstable     = np.array([int(np.sum(ev.real > 1e-8))]))
        self.lbl_status.setText(f'\u2705 Saved \u2192 {os.path.basename(folder)}')

    # -----------------------------------------------------------------
    def _show_help(self):
        h = QDialog(self); h.setWindowTitle('Jacobian Eigenspectrum — Theory')
        h.setWindowFlags(h.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        h.setStyleSheet(STYLE); h.setMinimumSize(640, 700)
        vl = QVBoxLayout(h); vl.setSpacing(8); vl.setContentsMargins(14,14,14,14)
        txt = QPlainTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet('QPlainTextEdit{background:#0a0c14;color:#c8d0e7;'
                          'font-family:"Courier New";font-size:12px;border:1px solid #1e2230;}')
        txt.setPlainText("""\
JACOBIAN EIGENSPECTRUM ANALYSIS
════════════════════════════════

WHAT IS IT?
───────────
The Jacobian method linearises the continuous-time LLE around the
simulated field and checks whether all perturbations decay.
It gives a full picture of the stability landscape — not just a
single number like FTLE, but the decay rate of every mode.

THE LLE IN θ-DOMAIN
────────────────────
∂ψ_m/∂t = (iΔ - L_m)ψ_m  -  i(D₂/2)D̂ψ_m
          - i∑_n H_{mn}ψ_n  +  i|ψ_m|²ψ_m  +  f_m

where D̂ is the circulant dispersion operator (eigenvalues μ²),
L_m = kin + kex·δ(m,ISite) is the loss, and f_m is the pump.

REAL/IMAGINARY SPLIT
────────────────────
Writing ψ_m = A_m + iB_m gives two coupled real equations:

  f₁ = dA/dt = -L·A - Δ·B + (D₂/2)D̂·B + Hᵣ·B + Hᵢ·A - (A²+B²)·B + E
  f₂ = dB/dt = -L·B + Δ·A - (D₂/2)D̂·A - Hᵣ·A + Hᵢ·B + (A²+B²)·A

JACOBIAN BLOCKS  (2·NL·NF × 2·NL·NF)
──────────────────────────────────────
J = [ J₁₁   J₁₂ ]    state vector u = [A.ravel(), B.ravel()]
    [ J₂₁   J₂₂ ]

J₁₁ = Hᵢ⊗I  -  diag(L)  -  diag(2AB)
J₁₂ = -Δ·I  +  I⊗Dsp  +  Hᵣ⊗I  -  diag(A²+3B²)
J₂₁ = +Δ·I  -  I⊗Dsp  -  Hᵣ⊗I  +  diag(3A²+B²)
J₂₂ = Hᵢ⊗I  -  diag(L)  +  diag(2AB)

HOW IT IS COMPUTED HERE
────────────────────────
The Jacobian is evaluated DIRECTLY at the simulation field
hist[:, :, step] — the same approach used by Sanzida et al.
No Newton solve is performed. This is fast and unambiguous.

Note: this evaluates J at the discrete-map fixed point, not the
continuous-time fixed point. All eigenvalues will have Re = -k
exactly (degenerate), confirming stability but without resolving
the individual mode structure.

EIGENVALUE CLASSIFICATION
──────────────────────────
  bulk (blue):         Re ≈ -k = -(kin+kex), Im spans ±|Δ_eff|
  outlier (orange):    most-damped mode,  Im = 0
  near-Goldstone (green): least-damped,  Im = 0  (soliton translation)

STABILITY CRITERION
────────────────────
  All Re(λ) < 0  →  STABLE   (all perturbations decay)
  Any  Re(λ) > 0  →  UNSTABLE (at least one mode grows)

RELATIONSHIP TO FTLE
────────────────────
For a stationary soliton, the FTLE converges to:
  λ_FTLE ≈ max Re(λ_J) × dt

where dt is the slow-time step. This provides a consistency check
between the two methods.
""")
        vl.addWidget(txt)
        btn_close = QPushButton('Close'); btn_close.clicked.connect(h.accept)
        br = QHBoxLayout(); br.addStretch(); br.addWidget(btn_close)
        vl.addLayout(br)
        h.exec_()

    def stop(self):
        self._stop.set()


# =====================================================================
# MONODROMY TAB  —  Floquet multipliers via finite-difference map Jacobian
# =====================================================================
class _MonodromyTabWidget(QWidget):
    """Monodromy (Floquet) tab.

    Algorithm (per the document):
      1. Estimate period N_T from the ESA of hist power — first non-DC peak.
      2. Pick ψ* = hist[:, :, seed_step].
      3. Build M column-by-column:  M[:,j] = (G^N_T(ψ* + ε·eⱼ) - ψ*) / ε
         where G^N_T means running the split-step stepper N_T times.
      4. Eigenvalues μ of M are Floquet multipliers.
         Stability: all |μ| ≤ 1.
    """
    def __init__(self, p, hist, parent_win=None):
        super().__init__()
        self._p    = p
        self._hist = hist
        self._stop = threading.Event()
        # Signal from UI to worker: "user confirmed the 3T oscilloscope looks
        # right, proceed with monodromy build". Cleared at each Run.
        self._continue = threading.Event()
        self._q    = queue.Queue()
        self._timer= QTimer(self); self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)
        self._last_ev   = None
        self._last_meta = {}

        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(6)

        lbl_title = QLabel('MONODROMY MATRIX  (FLOQUET MULTIPLIERS)')
        lbl_title.setFont(QFont('Courier New', 11, QFont.Bold))
        lbl_title.setStyleSheet(f'color:{ACCENT};')
        lay.addWidget(lbl_title)

        # ── Parameter row ─────────────────────────────────────────────
        param_row = QHBoxLayout()
        def _lbl(t):
            l = QLabel(t); l.setStyleSheet('color:#c8d0e7;font-size:12px;'); return l

        self.spn_step = QSpinBox()
        self.spn_step.setRange(0, max(0, hist.shape[2]-1))
        self.spn_step.setValue(hist.shape[2]-1)
        self.spn_step.setToolTip('Seed step ψ* on the periodic orbit')

        self.spn_T_guess = QSpinBox()
        self.spn_T_guess.setRange(1, 1000000)
        self.spn_T_guess.setValue(1300)
        self.spn_T_guess.setSingleStep(100)
        self.spn_T_guess.setToolTip(
            'Approximate period in steps from oscilloscope — shooting scan will refine it')

        self.spn_scan = QSpinBox()
        self.spn_scan.setRange(0, 10000)
        self.spn_scan.setValue(200)
        self.spn_scan.setSingleStep(50)
        self.spn_scan.setToolTip(
            'Shooting scan half-width in steps (0 = use T_guess directly, no scan)')

        self.spn_eps = QDoubleSpinBox()
        self.spn_eps.setDecimals(1); self.spn_eps.setRange(-14, -2)
        self.spn_eps.setValue(-10)
        self.spn_eps.setToolTip('Finite-difference step: 10^(this value)')

        param_row.addWidget(_lbl('Seed step:')); param_row.addWidget(self.spn_step)
        param_row.addSpacing(8)
        param_row.addWidget(_lbl('T_guess:')); param_row.addWidget(self.spn_T_guess)
        param_row.addSpacing(8)
        param_row.addWidget(_lbl('±scan:')); param_row.addWidget(self.spn_scan)
        param_row.addSpacing(8)
        param_row.addWidget(_lbl('log₁₀(ε):')); param_row.addWidget(self.spn_eps)

        # Operating point label
        def _sched_at(val, key, fallback):
            ch = p.get('schedule', {})
            arr = ch.get(key)
            if arr is not None and val < len(arr):
                return float(arr[val])
            return float(p.get(fallback, float('nan')))

        init_step = self.spn_step.value()
        self.lbl_op = QLabel(
            f'Δ = {_sched_at(init_step,"detuning","detuning"):.4f}   '
            f'F = {_sched_at(init_step,"F","F"):.4f}')
        self.lbl_op.setStyleSheet('color:#ff8c42;font-size:12px;font-style:italic;')
        def _on_step(val):
            self.lbl_op.setText(
                f'Δ = {_sched_at(val,"detuning","detuning"):.4f}   '
                f'F = {_sched_at(val,"F","F"):.4f}')
        self.spn_step.valueChanged.connect(_on_step)
        param_row.addSpacing(12); param_row.addWidget(self.lbl_op)
        param_row.addStretch()

        self.btn_run = QPushButton('▶  Run Monodromy')
        self.btn_run.setFixedHeight(26)
        self.btn_run.setStyleSheet(
            'QPushButton:enabled{color:#a8ff78;border-color:#a8ff78;}'
            'QPushButton:disabled{color:#4a5270;border-color:#2a3050;background:#141a28;}')
        self.btn_run.clicked.connect(self._launch)
        param_row.addWidget(self.btn_run)

        self.btn_stop_mono = QPushButton('■  Stop')
        self.btn_stop_mono.setFixedHeight(26)
        self.btn_stop_mono.setEnabled(False)
        self.btn_stop_mono.clicked.connect(lambda: self._stop.set())
        param_row.addWidget(self.btn_stop_mono)
        lay.addLayout(param_row)

        # ── Status + progress bar ──────────────────────────────────────
        self.lbl_status = QLabel('Ready.')
        self.lbl_status.setStyleSheet('color:#4a5270;font-size:12px;')
        lay.addWidget(self.lbl_status)

        self.prog_mono = QProgressBar()
        self.prog_mono.setRange(0, 100); self.prog_mono.setValue(0)
        self.prog_mono.setFixedHeight(4); self.prog_mono.setTextVisible(False)
        self.prog_mono.setStyleSheet(
            'QProgressBar{background:#0d0f1a;border:none;border-radius:2px;}'
            'QProgressBar::chunk{background:#a8ff78;border-radius:2px;}')
        self.prog_mono.setVisible(False)
        lay.addWidget(self.prog_mono)

        # ── Plots: left = full unit circle;
        #         right-top = |a_W|² oscilloscope over 3T;
        #         right-bot = zoom of Floquet multipliers near (1, 0) ──
        self.fig = Figure(facecolor=DARK_BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.10,
                                 wspace=0.35, hspace=0.45)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1, 1])
        # Full multiplier plot spans both rows of the left column
        self.ax_circ = self.fig.add_subplot(gs[:, 0])
        # Right column: top = oscilloscope, bottom = zoom near ν=1
        self.ax_osc  = self.fig.add_subplot(gs[0, 1])
        self.ax_zoom = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_circ, self.ax_osc, self.ax_zoom):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_circ.set_xlabel('Re(μ)', color=TEXT_DIM, fontsize=9)
        self.ax_circ.set_ylabel('Im(μ)', color=TEXT_DIM, fontsize=9)
        self.ax_circ.set_title('Floquet multipliers', color=TEXT_COL, fontsize=9, pad=3)
        self.ax_osc.set_xlabel('step', color=TEXT_DIM, fontsize=9)
        self.ax_osc.set_ylabel('|a_W|²', color=TEXT_DIM, fontsize=9)
        self.ax_osc.set_title('|a_W|² over 3T from ψ*', color=TEXT_COL, fontsize=9, pad=3)
        self.ax_zoom.set_xlabel('Re(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        self.ax_zoom.set_ylabel('Im(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        self.ax_zoom.set_title('zoom near ν=1', color=TEXT_COL, fontsize=9, pad=3)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas, stretch=1)

        # Navigation toolbar: interactive pan/zoom on the Monodromy plots.
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar_mono = NavigationToolbar2QT(self.canvas, self)
        self.toolbar_mono.setStyleSheet(
            'QToolBar{background:#141a28;border:1px solid #3a4560;spacing:2px;}'
            'QToolButton{background:#1e2538;color:#c8d0e7;border:1px solid #3a4560;'
            '  border-radius:2px;padding:3px;}'
            'QToolButton:hover{background:#2a3148;border-color:#00e5ff;}'
            'QToolButton:checked{background:#00e5ff;color:#0e1520;border-color:#00e5ff;}'
            'QLabel{color:#c8d0e7;font-size:11px;background:#141a28;}')
        lay.addWidget(self.toolbar_mono)

        self.lbl_result = QLabel('')
        self.lbl_result.setFont(QFont('Courier New', 11, QFont.Bold))
        lay.addWidget(self.lbl_result)

        self.lbl_period = QLabel('')
        self.lbl_period.setStyleSheet('color:#ff8c42;font-size:12px;font-family:"Courier New";')
        lay.addWidget(self.lbl_period)

        bot_row = QHBoxLayout()
        self.btn_save_mono = QPushButton('💾  Save multipliers')
        self.btn_save_mono.setFixedHeight(28)
        self.btn_save_mono.setEnabled(False)
        self.btn_save_mono.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_mono.clicked.connect(self._save)
        btn_help_mono = QPushButton('❓  Help')
        btn_help_mono.setFixedHeight(28)
        btn_help_mono.clicked.connect(self._show_help)
        bot_row.addStretch()
        bot_row.addWidget(self.btn_save_mono)
        bot_row.addWidget(btn_help_mono)
        lay.addLayout(bot_row)

    # -----------------------------------------------------------------
    def _detect_period(self, hist, OSite, dt):
        """Estimate N_T from the oscilloscope trace (total |a_W|^2 vs step).

        The breather period is the slow envelope of total power — visible in
        the oscilloscope bottom panel of SlowTimeWindow. This is NOT the ESA
        per-mode peak (which gives comb tooth beat frequency, much faster).

        Algorithm:
          1. Compute total |a_W|^2 per step (same as oscilloscope panel)
          2. FFT the de-meaned trace -> find first significant peak
          3. N_T = round(NS / peak_bin)

        Returns (N_T, omega_rep) in J units, or (1, 0.0) if stationary.
        """
        NS      = hist.shape[2]
        OSite_i = OSite - 1

        # Total spectral power per step: sum over all FSR modes of |a_W|^2
        # a_W = fft(a_T) -> |a_W|^2 summed = total intracavity power (Parseval)
        # This is exactly the oscilloscope trace
        power_vs_step = np.array([
            float(np.sum(np.abs(np.fft.fft(hist[OSite_i, :, i]))**2))
            for i in range(NS)
        ])

        # FFT of de-meaned power trace
        p_ac     = power_vs_step - power_vs_step.mean()
        spec     = np.abs(np.fft.rfft(p_ac))**2   # (NS//2+1,)
        # freq axis in J units (matching SlowTimeWindow ESA convention)
        freq_esa = np.fft.rfftfreq(NS, d=dt) * 2*np.pi   # [rad/step = J]

        if len(spec) < 2:
            return 1, 0.0

        # Skip DC, normalise
        spec_ac   = spec[1:]
        spec_norm = spec_ac / (spec_ac.max() + 1e-30)

        # Lowest-frequency bin above 1% threshold = fundamental breather freq
        sig = np.where(spec_norm >= 0.01)[0]
        if len(sig) == 0:
            return 1, 0.0   # flat power -> stationary soliton

        omega_rep = float(freq_esa[sig[0] + 1])   # +1 for skipped DC
        if omega_rep <= 0:
            return 1, 0.0

        N_T = max(1, round(2*np.pi / omega_rep))
        return N_T, omega_rep

    # -----------------------------------------------------------------
    def _launch(self):
        self._stop.clear()
        self._continue.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop_mono.setEnabled(True)
        self.btn_save_mono.setEnabled(False)
        self.lbl_result.setText('')
        self.prog_mono.setValue(0); self.prog_mono.setVisible(True)
        step = self.spn_step.value()
        eps  = 10 ** self.spn_eps.value()

        p    = self._p
        hist = self._hist
        q    = self._q

        def worker():
            try:
                NL       = p['NLattice']
                one_side = p['one_side_fsr']
                D2       = p['D2']
                kin      = p['kin'];  kex = p['kex']
                ISite    = p['ISite']; OSite = p['OSite']
                NF       = 2*one_side + 1
                NLNF     = NL * NF
                dt       = p['TStep']

                sched_ch = p.get('schedule', {})
                def _at(key, fallback):
                    arr = sched_ch.get(key)
                    if arr is not None and step < len(arr):
                        return float(arr[step])
                    return float(fallback)
                Delta = _at('detuning', p['detuning'])
                F_val = _at('F',        p['F'])

                # ── Build propagators — exact match of app stepper ────
                # Use build_propagators (full H, correct normalization)
                # so G(a) is identical to the simulation stepper.
                psi_val = p.get('psi', 0.0)
                ls      = p.get('linear_state', {})
                if ls.get('is_cyl') and build_hamiltonian is not None:
                    H_full = build_hamiltonian(
                        ls['h_str'], ls['nx0'], ls['ny0'],
                        ls['nx1'], ls['ny1'],
                        ls.get('j1', 0.3),
                        ls.get('phi_iqh0', np.pi/2), ls.get('phi_iqh1', np.pi/2),
                        ls.get('phi_aqh0', np.pi/4), ls.get('phi_aqh1', np.pi/4),
                        psi_val)
                    H = H_full[1:, 1:].copy()
                else:
                    H = p['H_mat'].copy()
                for k_h, v_h in p.get('heaters', {}).items():
                    n_h = int(k_h.split('_')[1]); H[n_h-1, n_h-1] += v_h

                DispMat, LM_full, FSRVec_bp, PumpFSR, NumFSRs_bp = build_disp_loss(
                    NL, one_side, D2, kin, kex, ISite, OSite)
                props_full, fc_full, _ = build_propagators(
                    H, DispMat, LM_full, Delta, dt, F_val, PumpFSR, ISite)
                # props_full: (NumFSRs, NL, NL), fc_full: (NumFSRs, NL)

                def G(a):
                    """One split-step iteration — identical to app stepper."""
                    a2 = a * np.exp(1j * np.abs(a)**2 * (dt/2))
                    a_W = np.fft.fft(a2, axis=1, norm='ortho')
                    a_W = np.einsum('mij,jm->im', props_full, a_W) + fc_full.T
                    at  = np.fft.ifft(a_W, axis=1, norm='ortho')
                    return at * np.exp(1j * np.abs(at)**2 * (dt/2))

                def G_N(a, N):
                    """Apply G exactly N times."""
                    for _ in range(N):
                        if self._stop.is_set():
                            return None
                        a = G(a)
                    return a

                # ── Seed ψ* from hist ─────────────────────────────────
                psi_star = hist[:, :, step].copy()   # (NL, NF) complex

                # ── Step 1: shooting scan to find N_T ────────────────
                T_guess  = self.spn_T_guess.value()
                scan_hw  = self.spn_scan.value()   # half-width

                if scan_hw == 0:
                    # User trusts their guess — use directly
                    N_T = T_guess
                    residual_scan = float('nan')
                    q.put(('status', f'Using N_T={N_T} directly (no scan)'))
                else:
                    # Coarse scan: step of 10 over ±scan_hw
                    coarse_step = max(1, scan_hw // 20)
                    N_vals = list(range(
                        max(1, T_guess - scan_hw),
                        T_guess + scan_hw + 1,
                        coarse_step))
                    q.put(('status',
                           f'Shooting scan: {len(N_vals)} trials in '
                           f'[{N_vals[0]}, {N_vals[-1]}] (step {coarse_step})…'))

                    # Evolve ψ* to each N and measure residual
                    # To avoid re-running from ψ* each time, we walk forward
                    # incrementally: store checkpoint every coarse_step steps
                    psi_scan = psi_star.copy()
                    step_count = 0
                    residuals_coarse = []
                    for N_try in N_vals:
                        if self._stop.is_set():
                            q.put(('stopped',)); return
                        steps_needed = N_try - step_count
                        psi_scan = G_N(psi_scan, steps_needed)
                        if psi_scan is None:
                            q.put(('stopped',)); return
                        step_count = N_try
                        res = float(np.linalg.norm(psi_scan - psi_star) /
                                    (np.linalg.norm(psi_star) + 1e-30))
                        residuals_coarse.append(res)

                    best_coarse = N_vals[int(np.argmin(residuals_coarse))]

                    # Fine scan: ±coarse_step around best coarse estimate, step 1
                    fine_range = range(
                        max(1, best_coarse - coarse_step),
                        best_coarse + coarse_step + 1)
                    q.put(('status',
                           f'Fine scan: [{fine_range.start}, {fine_range.stop-1}]…'))

                    # Walk from ψ* to fine_range.start
                    psi_fine = G_N(psi_star, fine_range.start)
                    if psi_fine is None:
                        q.put(('stopped',)); return
                    step_count = fine_range.start
                    residuals_fine = []
                    N_fine_vals = list(fine_range)
                    for N_try in N_fine_vals:
                        if self._stop.is_set():
                            q.put(('stopped',)); return
                        steps_needed = N_try - step_count
                        psi_fine = G_N(psi_fine, steps_needed)
                        if psi_fine is None:
                            q.put(('stopped',)); return
                        step_count = N_try
                        res = float(np.linalg.norm(psi_fine - psi_star) /
                                    (np.linalg.norm(psi_star) + 1e-30))
                        residuals_fine.append(res)

                    best_idx  = int(np.argmin(residuals_fine))
                    N_T       = N_fine_vals[best_idx]
                    residual_scan = residuals_fine[best_idx]
                    q.put(('status',
                           f'Shooting scan done: N_T={N_T}  '
                           f'residual={residual_scan:.3e}  '
                           f'(T_guess was {T_guess})'))

                if self._stop.is_set():
                    q.put(('stopped',)); return

                # ── Step 2: residual check / verify orbit ─────────────
                # psi_star already set above

                # Residual check: reuse shooting result if available
                if scan_hw > 0:
                    residual = residual_scan
                    q.put(('status',
                           f'N_T={N_T}  |G^T(ψ*)-ψ*|/|ψ*| = {residual:.3e}'))
                else:
                    q.put(('status', f'Verifying ψ* on orbit (evolving {N_T} steps)…'))
                    psi_check = G_N(psi_star, N_T)
                    if psi_check is None:
                        q.put(('stopped',)); return
                    residual = float(np.linalg.norm(psi_check - psi_star) /
                                     (np.linalg.norm(psi_star) + 1e-30))
                    q.put(('status',
                           f'N_T={N_T}  |G^T(ψ*)-ψ*|/|ψ*| = {residual:.3e}'))

                # ── Phase 1: evolve ψ* for 3T, record |a_W|², show to user ───
                # Do this BEFORE the expensive monodromy build. The 3T
                # oscilloscope is what tells the user whether N_T is correct.
                # If the period is wrong, the trace won't cleanly repeat —
                # better to find out now than after an hour of FD columns.
                # We ALSO time this 3T evolution — it's a calibration of the
                # per-period cost on the user's machine right now, which lets
                # us give an accurate ETA for the FD-column phase.
                q.put(('status', f'N_T={N_T}  — evolving ψ* for 3T for period check…'))
                N_3T    = 3 * N_T
                osc_pwr = np.zeros(N_3T + 1)
                a_osc   = psi_star.copy()
                osc_pwr[0] = float(np.sum(np.abs(
                    np.fft.fft(a_osc[OSite-1, :], norm='ortho'))**2))
                import time as _time
                _osc_t0 = _time.time()
                for i in range(N_3T):
                    if self._stop.is_set():
                        q.put(('stopped',)); return
                    a_osc = G(a_osc)
                    osc_pwr[i+1] = float(np.sum(np.abs(
                        np.fft.fft(a_osc[OSite-1, :], norm='ortho'))**2))
                _osc_elapsed = _time.time() - _osc_t0

                # Empirical per-period cost, measured JUST NOW on this machine
                # with these JAX/MKL settings and whatever else is competing
                # for CPU. Much more reliable than any hard-coded calibration.
                t_per_period  = _osc_elapsed / 3.0   # seconds per period
                N_cols_est    = 2 * NLNF
                # FD-column phase: each column = 1 full-period evolution
                t_fd_est      = N_cols_est * t_per_period
                # Eigenvalues phase: O(N³) scaling, calibrated to your Xeon
                # w5-2565X with MKL (see Jacobian ETA for derivation)
                t_eig_est     = 2.3 * (N_cols_est / 2050.0) ** 3
                t_total_est   = t_fd_est + t_eig_est
                def _fmt(s):
                    s = max(int(s), 0)
                    h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
                    if h: return f'{h}h {m}m {s}s'
                    if m: return f'{m}m {s}s'
                    return f'{s}s'
                _eta_str_fd  = _fmt(t_fd_est)
                _eta_str_eig = _fmt(t_eig_est)
                _eta_str_tot = _fmt(t_total_est)
                _end_est     = _time.strftime('%H:%M:%S',
                    _time.localtime(_time.time() + t_total_est))

                # Hand the oscilloscope trace to the UI and wait for the user.
                # The UI will either set _continue (proceed) or _stop (abort).
                self._continue.clear()
                q.put(('orbit_check', N_T, residual, osc_pwr,
                       t_per_period, t_fd_est, t_eig_est,
                       _eta_str_fd, _eta_str_eig, _eta_str_tot, _end_est,
                       N_cols_est))
                # Block until UI dispatches either continue or stop
                while not (self._continue.is_set() or self._stop.is_set()):
                    # poll every 100 ms; avoids busy-loop, stays responsive
                    if self._continue.wait(timeout=0.1):
                        break
                if self._stop.is_set():
                    q.put(('stopped',)); return

                q.put(('status',
                       f'N_T={N_T}  residual={residual:.3e}  '
                       f'— building monodromy…'))

                # Reference: G^N_T(ψ*) — needed for FD columns
                psi_check = G_N(psi_star, N_T)
                if psi_check is None:
                    q.put(('stopped',)); return

                # ── Step 3: build M column-by-column ──────────────────
                # M[:,j] = (G^N_T(ψ* + ε·eⱼ) - ψ*) / ε
                # state vector u = [Re(ψ*).ravel(), Im(ψ*).ravel()]
                u_star = np.concatenate([psi_star.real.ravel(), psi_star.imag.ravel()])
                # Reference: G^N_T(ψ*) — already computed above
                g0 = np.concatenate([psi_check.real.ravel(), psi_check.imag.ravel()])

                N_cols = 2 * NLNF
                Mono   = np.zeros((N_cols, N_cols), dtype=float)

                for j in range(N_cols):
                    if self._stop.is_set():
                        q.put(('stopped',)); return

                    u1    = u_star.copy(); u1[j] += eps
                    a1    = (u1[:NLNF] + 1j*u1[NLNF:]).reshape(NL, NF)
                    a1_T  = G_N(a1, N_T)
                    if a1_T is None:
                        q.put(('stopped',)); return
                    g1    = np.concatenate([a1_T.real.ravel(), a1_T.imag.ravel()])
                    Mono[:, j] = (g1 - g0) / eps

                    if j % max(1, N_cols // 50) == 0:
                        pct = int(100 * j / N_cols)
                        q.put(('prog', pct))
                        q.put(('status',
                               f'N_T={N_T}  residual={residual:.2e}  —  '                               f'FD column {j}/{N_cols}…'))

                # ── Step 4: eigenvalues ───────────────────────────────
                q.put(('status', 'Computing eigenvalues…'))
                ev = np.linalg.eigvals(Mono)

                omega_rep = float(2*np.pi / N_T) if N_T > 1 else 0.0

                q.put(('done', ev, Delta, F_val, N_T, omega_rep, residual, osc_pwr))

            except Exception as exc:
                import traceback; traceback.print_exc()
                q.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self._timer.start()

    # -----------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait(); kind = msg[0]
                if kind == 'status':
                    self.lbl_status.setText(msg[1])
                elif kind == 'prog':
                    self.prog_mono.setValue(msg[1])
                elif kind == 'orbit_check':
                    # Worker has detected N_T and produced the 3T oscilloscope.
                    # Show it to the user BEFORE building the monodromy matrix —
                    # if N_T is wrong the trace won't cleanly repeat, and we can
                    # abort before spending the big compute.
                    (_, N_T, residual, osc_pwr,
                     t_per_period, t_fd_est, t_eig_est,
                     eta_str_fd, eta_str_eig, eta_str_tot, end_est,
                     N_cols_est) = msg
                    # Remember timing info so the done-handler can show actual
                    # vs. estimated runtime when the build completes.
                    self._timing_estimate = dict(
                        t_per_period = t_per_period,
                        t_fd_est     = t_fd_est,
                        t_eig_est    = t_eig_est,
                        t_total_est  = t_fd_est + t_eig_est,
                        N_cols       = N_cols_est,
                    )
                    self._draw_osc_preview(N_T, residual, osc_pwr)
                    # Pause progress bar while we wait for the user
                    self.prog_mono.setVisible(False)
                    # The modal confirm dialog re-enters Qt's event loop. Stop
                    # our poll timer during that so _poll isn't called recursively
                    # on new messages while we're still processing this one.
                    self._timer.stop()
                    try:
                        self._ask_user_to_confirm(
                            N_T, residual,
                            eta_str_fd, eta_str_eig, eta_str_tot, end_est,
                            N_cols_est)
                    finally:
                        self._timer.start()
                    return   # exit this _poll call; next tick picks up new msgs
                elif kind == 'done':
                    _, ev, Delta, F_val, N_T, omega_rep, residual, osc_pwr = msg
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_mono.setEnabled(False)
                    self.prog_mono.setValue(100); self.prog_mono.setVisible(False)
                    self._last_ev   = ev
                    self._last_meta = dict(
                        step     = self.spn_step.value(),
                        Delta    = Delta, F = F_val,
                        N_T      = N_T,  omega_rep = omega_rep,
                        residual = residual,
                        D2       = float(self._p.get('D2',  float('nan'))),
                        kin      = float(self._p.get('kin', float('nan'))),
                        kex      = float(self._p.get('kex', float('nan'))),
                        NLattice = int(self._p.get('NLattice', 1)),
                        one_side = int(self._p.get('one_side_fsr', 128)),
                        dt       = float(self._p.get('TStep', float('nan'))))
                    self.btn_save_mono.setEnabled(True)
                    self._plot(ev, N_T, omega_rep, residual, osc_pwr)
                elif kind == 'stopped':
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_mono.setEnabled(False)
                    self.prog_mono.setVisible(False)
                    # Wipe any placeholder text on the left/zoom panels so they
                    # don't linger showing "Awaiting…" or "Building…" after a
                    # stop. Oscilloscope (if drawn) stays visible as diagnostic.
                    for ax in (self.ax_circ, self.ax_zoom):
                        ax.cla(); ax.set_facecolor(PANEL_BG)
                        ax.tick_params(colors=TEXT_COL, labelsize=8)
                        ax.grid(True, color=GRID_COL, lw=0.4)
                        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
                        ax.set_xticks([]); ax.set_yticks([])
                    self.canvas.draw_idle()
                    self.lbl_status.setText('Stopped.')
                elif kind == 'error':
                    self._timer.stop()
                    self.btn_run.setEnabled(True)
                    self.btn_stop_mono.setEnabled(False)
                    self.prog_mono.setVisible(False)
                    self.lbl_status.setText(f'Error: {msg[1]}')
        except queue.Empty:
            pass

    # -----------------------------------------------------------------
    def _draw_osc_preview(self, N_T, residual, osc_pwr):
        """Draw ONLY the 3T oscilloscope panel, leave the other two blank.
        Called when the worker finishes Phase 1 and asks the user to confirm
        the period before building the monodromy matrix."""
        # Clear all three panels, set titles to a "pending" state
        for ax in (self.ax_circ, self.ax_zoom):
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            ax.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.text(0.5, 0.5, 'Awaiting confirmation…',
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM, fontsize=10, fontstyle='italic')
            ax.set_xticks([]); ax.set_yticks([])

        # Draw the oscilloscope
        ax2 = self.ax_osc; ax2.cla()
        ax2.set_facecolor(PANEL_BG)
        ax2.tick_params(colors=TEXT_COL, labelsize=8)
        ax2.grid(True, color=GRID_COL, lw=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor('#3a4560')
        steps_arr = np.arange(len(osc_pwr))
        ax2.plot(steps_arr, osc_pwr, color='#c8d0e7', lw=1.0)
        # Period boundaries — these are where the trace should visibly repeat
        for k in range(1, 4):
            ax2.axvline(k * N_T, color='#ff8c42', lw=0.8, ls='--', alpha=0.7)
        ax2.set_xlabel('step', color=TEXT_DIM, fontsize=9)
        ax2.set_ylabel('|a_W|²', color=TEXT_DIM, fontsize=9)
        ax2.set_title(f'|a_W|² over 3T from ψ*  (N_T = {N_T} steps)  —  '
                      f'does the trace repeat at the orange dashes?',
                      color='#ffcc00', fontsize=9, pad=3)
        ax2.set_xlim(0, 3 * N_T)
        self.canvas.draw_idle()

    # -----------------------------------------------------------------
    def _ask_user_to_confirm(self, N_T, residual,
                             eta_str_fd='', eta_str_eig='', eta_str_tot='',
                             end_est='', N_cols_est=0):
        """Modal dialog: user inspects the oscilloscope and decides whether to
        proceed with the expensive monodromy build."""
        dlg = QMessageBox(self)
        dlg.setWindowTitle('Confirm period')
        dlg.setStyleSheet(STYLE)
        dlg.setIcon(QMessageBox.Question)

        eta_block = ''
        if eta_str_tot:
            eta_block = (
                f'\n\n── Estimated cost on this machine ──\n'
                f'Monodromy size: {N_cols_est}×{N_cols_est}\n'
                f'FD-column phase:  ~{eta_str_fd}\n'
                f'Eigenvalues:      ~{eta_str_eig}\n'
                f'Total:            ~{eta_str_tot}  '
                f'(≈ done {end_est})\n')

        dlg.setText(
            f'Detected period N_T = {N_T} steps\n'
            f'Shooting residual |G^T(ψ*) − ψ*| / |ψ*| = {residual:.3e}\n\n'
            f'Look at the oscilloscope panel: the trace should repeat exactly '
            f'at each orange dashed line. If the peaks and troughs line up '
            f'across the three periods, N_T is correct and the monodromy '
            f'build will be meaningful.\n\n'
            f'If the trace drifts, has a different shape in each period, or '
            f'looks non-periodic, stop and retry with a different T_guess or '
            f'a longer slow-time evolution so ψ* is truly on the limit cycle.'
            + eta_block)
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        yes_btn = dlg.button(QMessageBox.Yes)
        yes_btn.setText('Looks right — build monodromy')
        no_btn  = dlg.button(QMessageBox.No)
        no_btn.setText('Abort')

        self.lbl_status.setText(
            f'N_T = {N_T}  —  please inspect the 3T oscilloscope '
            f'and confirm the period')

        # exec_() is modal; the worker is blocked on self._continue / self._stop
        # and will pick up whichever we set below.
        result = dlg.exec_()
        if result == QMessageBox.Yes:
            # Replace "Awaiting confirmation…" placeholder with "Building…"
            # so the user can distinguish "user action needed" from "compute in progress".
            build_msg = 'Building monodromy matrix…'
            if eta_str_tot:
                build_msg += f'\n(est ~{eta_str_tot}, ≈ done {end_est})'
            for ax in (self.ax_circ, self.ax_zoom):
                ax.cla(); ax.set_facecolor(PANEL_BG)
                ax.tick_params(colors=TEXT_COL, labelsize=8)
                ax.grid(True, color=GRID_COL, lw=0.4)
                for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
                ax.text(0.5, 0.5, build_msg,
                        transform=ax.transAxes, ha='center', va='center',
                        color=TEXT_DIM, fontsize=10, fontstyle='italic')
                ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            # Re-show progress bar for the build phase
            self.prog_mono.setVisible(True)
            self.prog_mono.setValue(0)
            _build_status = f'N_T = {N_T}  confirmed  —  building monodromy matrix…'
            if eta_str_tot:
                _build_status += f'  (est ~{eta_str_tot})'
            self.lbl_status.setText(_build_status)
            self._continue.set()
        else:
            # User aborted: clear placeholder text so the UI doesn't look stuck.
            # Leave the oscilloscope visible (useful diagnostic for next attempt).
            for ax in (self.ax_circ, self.ax_zoom):
                ax.cla(); ax.set_facecolor(PANEL_BG)
                ax.tick_params(colors=TEXT_COL, labelsize=8)
                ax.grid(True, color=GRID_COL, lw=0.4)
                for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
                ax.text(0.5, 0.5, 'Aborted — adjust T_guess and run again',
                        transform=ax.transAxes, ha='center', va='center',
                        color='#ff4a6e', fontsize=10, fontstyle='italic')
                ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            self.lbl_status.setText('Aborted by user — retry with a different T_guess.')
            self._stop.set()

    # -----------------------------------------------------------------
    def _plot(self, ev, N_T, omega_rep, residual, osc_pwr):
        dt      = float(self._last_meta.get('dt',  0.1))
        kin_val = float(self._last_meta.get('kin', float('nan')))
        kex_val = float(self._last_meta.get('kex', float('nan')))
        k_tot   = kin_val + kex_val
        T_phys  = N_T * dt

        # ── T-normalized multipliers: ν = μ^(1/T_phys) = exp(λ) ─────────────
        # This makes the display T-independent: any reference ring (κ, bulk loss,
        # supermode losses) lands on exp(-κ) regardless of the period T. A
        # multiplier with |ν|>1 is unstable irrespective of how T was chosen.
        # Stability verdicts are identical for raw vs normalized multipliers
        # (|μ|>1 ⟺ |ν|>1), but the normalized values make marginal modes more
        # visible and make plots comparable across different T values.
        #
        # Note on phase: arg(ν) = arg(μ)/T is bandlimited to (-π/T, +π/T].
        # This is NOT a normalization artefact — it's intrinsic to the period-T
        # stroboscopic map. Perturbations oscillating faster than ±π/T rad/J
        # alias in BOTH raw and normalized views. For stationary fixed points
        # (N_T=1), T_phys=dt so the bandlimit ±π/dt is typically far above any
        # physical mode frequency; aliasing is only a concern for long periods.
        # (T_phys>0 always in our setup — dt>0 and N_T>=1.)
        ev_n   = ev.astype(np.complex128) ** (1.0 / T_phys)
        mags_n = np.abs(ev_n)
        mags   = np.abs(ev)                  # raw |μ|, used for save-file stats
        r_bulk = float(np.exp(-k_tot))       # reference ring at exp(-κ_tot)

        n_unstable = int(np.sum(mags_n > 1 + 1e-6))
        verdict = 'STABLE' if n_unstable == 0 else 'UNSTABLE'
        color   = '#a8ff78' if n_unstable == 0 else '#ff4a6e'

        def pt_color(m):
            if m > 1 + 1e-6: return '#ff4a6e'
            if m > 1 - 1e-4: return '#a8ff78'
            return '#378ADD'
        colors = [pt_color(m) for m in mags_n]

        # ── Left: unit circle with Floquet multipliers ────────────────
        ax = self.ax_circ; ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        th = np.linspace(0, 2*np.pi, 300)
        # Unit circle
        ax.plot(np.cos(th), np.sin(th), color='white', lw=0.8, ls='--', alpha=0.5)

        # κ_σ circles: one per supermode of H.
        # Prefer a fresh recompute from H_mat because the linear_state cache may
        # be stale (saved with an older buggy Linear.py). Fall back to the cache
        # only when H_mat or loss values aren't available.
        kappa_sigma = None
        H_mat = self._p.get('H_mat')
        ISite = self._p.get('ISite', 1)
        OSite = self._p.get('OSite', 1)
        if H_mat is not None and not np.isnan(kin_val) and not np.isnan(kex_val):
            try:
                _, V_H = np.linalg.eigh(H_mat)
                # κ_σ = κ_in + Σ_{s ∈ port_sites} κ_ex · |⟨σ|s⟩|²
                # Use unique port sites so ISite==OSite isn't double-counted.
                kappa_sigma = np.full(V_H.shape[1], kin_val, dtype=float)
                for port in {ISite, OSite}:
                    kappa_sigma += kex_val * np.abs(V_H[port-1, :])**2
            except Exception:
                pass

        if kappa_sigma is None:
            ls_state = self._p.get('linear_state', {})
            sp_state = ls_state.get('spectrum', {}) if ls_state else {}
            if sp_state:
                kappa_sigma = sp_state.get('kappa_sigma')

        if kappa_sigma is not None:
            for k_s in kappa_sigma:
                r_s = float(np.exp(-k_s))        # T-normalized: exp(-κ_s)
                ax.plot(r_s*np.cos(th), r_s*np.sin(th),
                        color='#ff8c42', lw=1.0, ls=':', alpha=0.8, zorder=3)

        # Highlight bulk reference circle exp(-κ_tot) (T-normalized)
        ax.plot(r_bulk * np.cos(th), r_bulk * np.sin(th),
                color='#ff8c42', lw=1.2, ls=':', alpha=0.95, zorder=3,
                label=f'exp(-κ)={r_bulk:.4f}')
        ax.scatter(ev_n.real, ev_n.imag, c=colors, s=6, linewidths=0, zorder=2)
        ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_COL, loc='upper left')
        ax.set_xlabel('Re(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        ax.set_ylabel('Im(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        period_str = f'T={N_T} steps' if omega_rep > 0 else 'stationary'
        ax.set_title(f'Floquet multipliers  [{period_str}]  —  {verdict}',
                     color=color, fontsize=9, pad=3)
        ax.set_aspect('equal')

        # ── Right: |a_W|² over 3T from ψ* ────────────────────────────
        ax2 = self.ax_osc; ax2.cla()
        ax2.set_facecolor(PANEL_BG)
        ax2.tick_params(colors=TEXT_COL, labelsize=8)
        ax2.grid(True, color=GRID_COL, lw=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor('#3a4560')
        steps_arr = np.arange(len(osc_pwr))
        ax2.plot(steps_arr, osc_pwr, color='#c8d0e7', lw=1.0)
        # Mark period boundaries with vertical lines
        for k in range(1, 4):
            ax2.axvline(k * N_T, color='#ff8c42', lw=0.8, ls='--', alpha=0.7)
        ax2.set_xlabel('step', color=TEXT_DIM, fontsize=9)
        ax2.set_ylabel('|a_W|²', color=TEXT_DIM, fontsize=9)
        ax2.set_title(f'|a_W|² over 3T from ψ*  (T={N_T} steps)',
                      color=TEXT_COL, fontsize=9, pad=3)
        ax2.set_xlim(0, 3 * N_T)

        # ── Right-bottom: zoom of ν-plane around (1, 0) ──────────────
        # Shows the Goldstone / marginal modes clearly. Zoom window is
        # adapted to the data: use 3x the distance from 1 of the farthest
        # near-unit mode, with a sensible floor. No aspect='equal' so the
        # panel fills the available column width (same as the oscilloscope).
        ax3 = self.ax_zoom; ax3.cla()
        ax3.set_facecolor(PANEL_BG)
        ax3.tick_params(colors=TEXT_COL, labelsize=8)
        ax3.grid(True, color=GRID_COL, lw=0.4)
        for sp in ax3.spines.values(): sp.set_edgecolor('#3a4560')
        # Scatter first, reference lines on top
        ax3.scatter(ev_n.real, ev_n.imag, c=colors, s=18, linewidths=0, zorder=2)
        # unit circle (a small arc will appear in the zoom window)
        ax3.plot(np.cos(th), np.sin(th), color='white', lw=0.8, ls='--', alpha=0.7, zorder=3)
        # reference ν=1 cross-hairs
        ax3.axvline(1.0, color='white', lw=0.5, alpha=0.4, zorder=3)
        ax3.axhline(0.0, color='white', lw=0.5, alpha=0.4, zorder=3)
        # κ_σ reference circles — drawn on top so they're visible through the scatter
        if kappa_sigma is not None:
            for k_s in kappa_sigma:
                r_s = float(np.exp(-k_s))
                ax3.plot(r_s*np.cos(th), r_s*np.sin(th),
                         color='#ff8c42', lw=1.0, ls=':', alpha=0.9, zorder=4)
        # Bulk κ_tot reference circle
        ax3.plot(r_bulk*np.cos(th), r_bulk*np.sin(th),
                 color='#ff8c42', lw=1.2, ls=':', alpha=1.0, zorder=4)
        # Adaptive zoom window around (1, 0)
        dist_to_one = np.abs(ev_n - 1.0)
        sorted_d = np.sort(dist_to_one)
        scale_hint = sorted_d[min(5, len(sorted_d)-1)]
        half_w = max(3.0 * scale_hint, 1e-4)      # floor so plot is never empty
        half_w = min(half_w, 0.5)                  # cap so we don't zoom out to full circle
        ax3.set_xlim(1 - half_w, 1 + half_w)
        ax3.set_ylim(-half_w, +half_w)
        # NOTE: no set_aspect('equal') — let the axes fill the column width
        ax3.set_xlabel('Re(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        ax3.set_ylabel('Im(μ^{1/T})', color=TEXT_DIM, fontsize=9)
        ax3.set_title(f'zoom near ν=1  (±{half_w:.1e})',
                      color=TEXT_COL, fontsize=9, pad=3)

        self.canvas.draw_idle()

        # Period info label
        if omega_rep > 0:
            self.lbl_period.setText(
                f'T = {N_T} steps  =  {T_phys:.3f} / J   '
                f'ω_rep = {omega_rep:.5f} J   '
                f'|G^T(ψ*)-ψ*|/|ψ*| = {residual:.3e}   '
                f'exp(-κ_tot) = {r_bulk:.6f}   '
                f'[raw: exp(-κ_tot·T) = {float(np.exp(-k_tot*T_phys)):.6f}]')
        else:
            self.lbl_period.setText('Stationary soliton  (N_T = 1)')

        self.lbl_status.setText(
            f'|ν|=|μ^(1/T)| ∈ [{mags_n.min():.6f}, {mags_n.max():.6f}]  |  '
            f'{n_unstable} unstable  |  max|ν| = {mags_n.max():.8f}  '
            f'[raw max|μ| = {mags.max():.6g}]')
        self.lbl_result.setText(f'max|ν| = {mags_n.max():.8f}   →   {verdict}')
        self.lbl_result.setStyleSheet(f'color:{color};font-size:12px;font-weight:bold;')

    # -----------------------------------------------------------------
    def _save(self):
        if self._last_ev is None:
            return
        folder = self._p.get('session_folder')
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)
        m    = self._last_meta
        _flt = lambda v: f'{v:.4g}'.replace('.', 'p').replace('-', 'm')
        base = (f'monodromy_step{m["step"]}_NT{m["N_T"]}'
                f'_D{_flt(m["Delta"])}_F{_flt(m["F"])}'
                f'_res{_flt(m["residual"])}')

        # ── PNG + SVG ─────────────────────────────────────────────────
        for fmt, dpi in [('png', 200), ('svg', 150)]:
            buf = io.BytesIO()
            self.fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            with open(os.path.join(folder, f'{base}.{fmt}'), 'wb') as fh:
                fh.write(buf.getvalue())

        # ── NPZ ───────────────────────────────────────────────────────
        ev       = self._last_ev
        T_phys_s = float(m['N_T']) * float(m['dt'])
        # T-normalized multipliers ν = μ^(1/T) = exp(λ)
        ev_norm  = ev.astype(np.complex128) ** (1.0 / T_phys_s) if T_phys_s > 0 else ev
        path = os.path.join(folder, f'{base}.npz')
        np.savez(path,
                 multipliers_re  = ev.real,
                 multipliers_im  = ev.imag,
                 multipliers_abs = np.abs(ev),
                 multipliers_norm_re  = ev_norm.real,   # ν = μ^(1/T)
                 multipliers_norm_im  = ev_norm.imag,
                 multipliers_norm_abs = np.abs(ev_norm),
                 step     = np.array([m['step']]),
                 N_T      = np.array([m['N_T']]),
                 omega_rep = np.array([m['omega_rep']]),
                 residual = np.array([m['residual']]),
                 Delta    = np.array([m['Delta']]),
                 F        = np.array([m['F']]),
                 D2       = np.array([m['D2']]),
                 kin      = np.array([m['kin']]),
                 kex      = np.array([m['kex']]),
                 dt       = np.array([m['dt']]),
                 NLattice = np.array([m['NLattice']]),
                 one_side_fsr = np.array([m['one_side']]),
                 max_abs      = np.array([float(np.abs(ev).max())]),
                 max_abs_norm = np.array([float(np.abs(ev_norm).max())]),
                 n_unstable = np.array([int(np.sum(np.abs(ev) > 1+1e-6))]))
        self.lbl_status.setText(f'✅ Saved → {os.path.basename(folder)}')

    # -----------------------------------------------------------------
    def _show_help(self):
        h = QDialog(self); h.setWindowTitle('Monodromy — Floquet Theory')
        h.setWindowFlags(h.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        h.setStyleSheet(STYLE); h.setMinimumSize(640, 700)
        vl = QVBoxLayout(h); vl.setSpacing(8); vl.setContentsMargins(14,14,14,14)
        txt = QPlainTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet('QPlainTextEdit{background:#0a0c14;color:#c8d0e7;'
                          'font-family:"Courier New";font-size:12px;border:1px solid #1e2230;}')
        txt.setPlainText("""MONODROMY MATRIX  —  FLOQUET MULTIPLIERS
═════════════════════════════════════════

WHAT IT IS
──────────
The Floquet / monodromy approach handles PERIODIC solutions
(breathers) that the Jacobian eigenspectrum cannot capture.
A breather satisfies ψ*(t + T) = ψ*(t) — it returns to itself
after period T, but is never stationary.

THE ONE-PERIOD MAP  F = G^N_T
──────────────────────────────
G is one split-step iteration (dt). The one-period map is:

    F(ψ) = G applied N_T times   where N_T = round(T / dt)

ψ* is a FIXED POINT of F:  F(ψ*) = ψ*

The monodromy matrix is the Jacobian of F at ψ*:

    M = ∂F/∂ψ|_{ψ=ψ*}

Its eigenvalues μ are the Floquet multipliers.
The Floquet exponents λ satisfy  μ = exp(λ T).

STABILITY CRITERION
────────────────────
  All |μ| < 1  →  STABLE   (perturbations decay each period)
  Any  |μ| > 1  →  UNSTABLE (at least one mode grows)

Special values:
  μ = 1 exactly:  continuous symmetry (e.g., translational invariance)
  |μ| = 1, μ ≠ 1: marginal / bifurcation point

PERIOD DETECTION — SHOOTING SCAN
──────────────────────────────────
The period T cannot be found reliably from the ESA spectrum.
To resolve T = 1300 steps to ±10 step accuracy spectrally,
you would need ~1.7 million slow-time snapshots in hist.

Instead, a shooting scan is used:

  1. Enter T_guess from the oscilloscope (visual estimate).
  2. Coarse scan: evaluate ||G^N(ψ*) - ψ*|| for N in
     [T_guess - ±scan, T_guess + ±scan] with step size
     ±scan/20. Find the minimum.
  3. Fine scan: step 1 around the coarse minimum.
  4. N_T = argmin of the residual.

Set ±scan = 0 to use T_guess directly without scanning.

DOES USING 2T INSTEAD OF T WORK?
──────────────────────────────────
Yes — with caveats. The 2T monodromy eigenvalues are μᵢ²
where μᵢ are the true period-T Floquet multipliers. So:
  |μ²| < 1  iff  |μ| < 1
The stability verdict is identical. But the angles double,
so you cannot read off the true Floquet exponents from a
2T computation. Use T if you want the actual spectrum.

QUALITY CHECK
─────────────
After detecting N_T, we verify:

  |G^N_T(ψ*) - ψ*| / |ψ*|  (shown in status bar)

Small residual (< 0.01) confirms ψ* is truly on the orbit.
Large residual means the hist hasn't converged to a limit cycle
— run more steps and try again.

NUMERICAL CONSTRUCTION
───────────────────────
M[:,j] = ( F(ψ* + ε·eⱼ) - F(ψ*) ) / ε
       = ( G^N_T(ψ* + ε·eⱼ) - ψ* ) / ε

Each column requires N_T stepper calls.
Total: 2·NL·NF columns × N_T steps each.

PARAMETERS
──────────
Seed step:  index into hist to use as ψ*. Any step where the
            field is on the limit cycle works. Last step is fine.
log₁₀(ε):  FD perturbation. ε = 10⁻⁷ is recommended.
            Too large: nonlinear contamination.
            Too small: floating-point noise.
""")
        vl.addWidget(txt)
        btn_close = QPushButton('Close'); btn_close.clicked.connect(h.accept)
        br = QHBoxLayout(); br.addStretch(); br.addWidget(btn_close)
        vl.addLayout(br)
        h.exec_()

    # -----------------------------------------------------------------
    def stop(self):
        self._stop.set()

# =====================================================================
# LYAPUNOV WINDOW
# =====================================================================
class LyapunovWindow(QMainWindow):
    def __init__(self, p, hist, T_steps, K, N_trans, eps, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Lyapunov Analysis')
        self.setMinimumSize(700, 500)
        self.setStyleSheet(STYLE)

        self._p       = p
        self._hist    = hist
        self._T       = T_steps
        self._K       = K
        self._N_trans = N_trans
        self._eps     = eps
        self._stop    = threading.Event()
        self._q       = queue.Queue()
        self._timer   = QTimer(self); self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setSpacing(6); root.setContentsMargins(8,8,8,8)

        self.lbl_title = QLabel('FINITE-TIME LYAPUNOV EXPONENT')
        self.lbl_title.setFont(QFont('Courier New', 11, QFont.Bold))
        self.lbl_title.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(self.lbl_title)

        # Plot: λ(t) convergence curve
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.fig    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL_BG)
        self.ax.set_xlabel('Cycle', color=TEXT_DIM, fontsize=9)
        self.ax.set_ylabel('λ(t)', color=TEXT_DIM, fontsize=9)
        self.ax.set_title('Running FTLE estimate', color=TEXT_COL, fontsize=10, pad=4)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')
        root.addWidget(self.canvas, stretch=1)

        # Progress + status
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.lbl_status = QLabel('Ready')
        self.lbl_status.setStyleSheet('color:#4a5270;font-size:12px;')
        root.addWidget(self.prog)
        root.addWidget(self.lbl_status)

        # Stop + Save buttons
        btn_row = QHBoxLayout()
        self.btn_stop = QPushButton('■  Stop')
        self.btn_stop.setFixedHeight(28)
        self.btn_stop.clicked.connect(self._stop.set)
        self.btn_save_lyap = QPushButton('💾  Save')
        self.btn_save_lyap.setFixedHeight(28)
        self.btn_save_lyap.setEnabled(False)
        self.btn_save_lyap.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_save_lyap.clicked.connect(self._save)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_save_lyap)
        root.addLayout(btn_row)

    def run(self):
        """Start FTLE computation in background thread."""
        p    = self._p
        hist = self._hist
        T    = self._T; K = self._K; N_trans = self._N_trans; eps = self._eps

        # Build propagator from params
        NL      = p['NLattice']
        one_side = p['one_side_fsr']
        D2      = p['D2'];  kin = p['kin'];  kex = p['kex']
        ISite   = p['ISite']; OSite = p['OSite']
        dt      = p['TStep']
        F       = p['F'];   detuning = p['detuning']

        psi_val = p.get('psi', 0.0)
        ls      = p.get('linear_state', {})
        if ls.get('is_cyl') and build_hamiltonian is not None:
            H_full = build_hamiltonian(
                ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2), ls.get('phi_iqh1', np.pi/2),
                ls.get('phi_aqh0', np.pi/4), ls.get('phi_aqh1', np.pi/4), psi_val)
            H = H_full[1:, 1:].copy()
        else:
            H = p['H_mat'].copy()
        for k, v in p.get('heaters', {}).items():
            n = int(k.split('_')[1]); H[n-1, n-1] += v

        _q = self._q

        def worker():
            try:
                import time
                DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
                    NL, one_side, D2, kin, kex, ISite, OSite)
                props, fc_arr, _ = build_propagators(
                    H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                props_j = jnp.array(props); fc_j = jnp.array(fc_arr)
                stepper = get_stepper(1)

                # Seed Psi1 from last frame of hist
                Psi1 = jnp.array(hist[:, :, -1])

                # Initial perturbation
                rng  = np.random.default_rng(42)
                delt = rng.standard_normal(Psi1.shape) + 1j*rng.standard_normal(Psi1.shape)
                delt = delt / np.linalg.norm(delt) * eps
                Psi2 = Psi1 + jnp.array(delt)

                N_total = N_trans + T // K
                log_sum = 0.0; cycle = 0
                running = []   # (cycle_idx, lambda_running)

                t0 = time.time()
                for i in range(N_total):
                    if self._stop.is_set():
                        _q.put(('stopped', running)); return

                    # Evolve K steps
                    for _ in range(K):
                        Psi1 = stepper(Psi1, props_j, fc_j, dt)
                        Psi2 = stepper(Psi2, props_j, fc_j, dt)

                    # Measure separation
                    diff = np.array(Psi2 - Psi1)
                    d    = np.linalg.norm(diff)
                    if d == 0:
                        d = 1e-300

                    # Renormalize Psi2
                    Psi2 = Psi1 + jnp.array(eps * diff / d)

                    if i >= N_trans:
                        cycle += 1
                        log_sum += np.log(d / eps)
                        lam = log_sum / (cycle * K)
                        running.append((cycle, lam))
                        elapsed = time.time() - t0
                        _q.put(('progress', cycle, T // K, lam, elapsed))

                _q.put(('done', running))

            except Exception as exc:
                import traceback
                print('LYAPUNOV WORKER ERROR:\n' + traceback.format_exc())
                _q.put(('stopped', []))

        threading.Thread(target=worker, daemon=True).start()
        self._timer.start()

    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait(); kind = msg[0]
                if kind == 'progress':
                    _, cycle, N, lam, elapsed = msg
                    pct = int(100 * cycle / N)
                    self.prog.setValue(pct)
                    rem = elapsed / cycle * (N - cycle) if cycle > 0 else 0
                    em, es = divmod(int(rem), 60)
                    eh, em = divmod(em, 60)
                    eta_str = f'{eh}h{em:02d}m' if eh else f'{em}m{es:02d}s'
                    self.lbl_status.setText(
                        f'Cycle {cycle}/{N}  |  λ = {lam:.6f}  |  ETA {eta_str}')
                    self._update_plot(lam)
                elif kind in ('done', 'stopped'):
                    _, running = msg
                    self._timer.stop()
                    self.btn_stop.setEnabled(False)
                    if running:
                        lam_final = running[-1][1]
                        self._finish(lam_final, kind == 'stopped')
        except queue.Empty:
            pass

    def _update_plot(self, lam_current):
        """Append current lambda and redraw convergence curve."""
        if not hasattr(self, '_running_data'):
            self._running_data = []
        self._running_data.append(lam_current)
        cycles = list(range(1, len(self._running_data) + 1))

        self.ax.cla()
        self.ax.set_facecolor(PANEL_BG)
        self.ax.plot(cycles, self._running_data, color=ACCENT, lw=1.2)
        self.ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        self.ax.set_xlabel('Cycle', color=TEXT_DIM, fontsize=9)
        self.ax.set_ylabel('λ(t)', color=TEXT_DIM, fontsize=9)
        self.ax.set_title(f'Running FTLE  |  λ = {lam_current:.6f}',
                          color=TEXT_COL, fontsize=10, pad=4)
        self.ax.tick_params(colors=TEXT_COL, labelsize=8)
        self.ax.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax.spines.values(): sp.set_edgecolor('#3a4560')
        self.canvas.draw_idle()

    def _finish(self, lam_final, stopped):
        self.prog.setValue(100 if not stopped else self.prog.value())
        chaos = lam_final > 0
        sign  = '> 0  →  CHAOTIC' if chaos else '< 0  →  STABLE'
        color = '#ff4a6e' if chaos else '#a8ff78'
        self.lbl_title.setText(f'λ = {lam_final:.6f}   ({sign})')
        self.lbl_title.setStyleSheet(f'color:{color};font-size:13px;font-weight:bold;')
        status = 'Stopped' if stopped else 'Done'
        self.lbl_status.setText(f'{status}  |  λ_final = {lam_final:.6f}  |  {"Chaotic" if chaos else "Stable"}')
        self.ax.set_title(f'FTLE  λ = {lam_final:.6f}  —  {"CHAOTIC" if chaos else "STABLE"}',
                          color=color, fontsize=10, pad=4)
        self.canvas.draw_idle()
        self.btn_save_lyap.setEnabled(True)

    def _save(self):
        folder = self._p.get('session_folder')
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)

        lam_final = self._running_data[-1] if hasattr(self, '_running_data') and self._running_data else None
        chaos     = lam_final is not None and lam_final > 0
        label     = 'chaotic' if chaos else 'stable'
        _flt      = lambda v: f'{v:.4g}'.replace('.', 'p').replace('-', 'm')

        base = f'lyapunov_T{self._T}_K{self._K}_{label}'
        if lam_final is not None:
            base += f'_lam{_flt(lam_final)}'

        # PNG + SVG of convergence plot
        for fmt, dpi in [('png', 200), ('svg', 150)]:
            buf = io.BytesIO()
            self.fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            with open(os.path.join(folder, f'{base}.{fmt}'), 'wb') as fh:
                fh.write(buf.getvalue())

        # NPZ: running lambda curve + params
        if hasattr(self, '_running_data') and self._running_data:
            np.savez(os.path.join(folder, f'{base}.npz'),
                     lambda_running = np.array(self._running_data),
                     lambda_final   = np.array([lam_final]),
                     T              = np.array([self._T]),
                     K              = np.array([self._K]),
                     N_trans        = np.array([self._N_trans]),
                     eps            = np.array([self._eps]),
                     detuning       = np.array([self._p['detuning']]),
                     F              = np.array([self._p['F']]),
                     D2             = np.array([self._p['D2']]))

        self.lbl_status.setText(f'✅ Saved → {os.path.basename(folder)}')


# =====================================================================
# SLOW TIME WINDOW (continued methods — same class via monkey-patching)
# =====================================================================
class _SlowTimeWindowMethods:
    # ── [EXPERIMENTAL] Modal decomposition ────────────────────────────
    def _on_modal(self):
        """Open modal decomposition window with Start/Stop/Save/Close."""
        if self._hist is None: return
        p  = self._p
        ls = p.get('linear_state', {})

        # ── Build popup window ─────────────────────────────────────────
        win = QDialog(self)
        win.setWindowTitle('Modal Decomposition')
        win.setWindowFlags(win.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        win.setStyleSheet(STYLE)
        win.setMinimumSize(1120, 780)
        vl = QVBoxLayout(win); vl.setSpacing(4); vl.setContentsMargins(8,8,8,8)

        hdr = QLabel('MODAL DECOMPOSITION')
        hdr.setFont(QFont('Courier New', 10, QFont.Bold))
        hdr.setStyleSheet('color:#cc88ff;')
        vl.addWidget(hdr)

        self._modal_status = QLabel('Press  ▶ Start Analysis  to compute.')
        self._modal_status.setStyleSheet('color:#4a5270;font-size:12px;')
        vl.addWidget(self._modal_status)

        # Figure
        fig = Figure(facecolor=DARK_BG, tight_layout=False)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.09,
                            hspace=0.55, wspace=0.4)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(canvas, stretch=1)

        # ── Draw placeholder layout immediately so window isn't blank ──
        # The placeholder mirrors the post-Start-Analysis layout exactly
        # (same gridspec, same axis styling, same labels/ticks) so the
        # window's geometry doesn't shift when analysis completes.
        # Only differences: panels have no data drawn, and wide panels
        # carry a centred "Press ▶ Start Analysis" hint.
        def _draw_placeholder(msg='Press  ▶ Start Analysis  to begin.'):
            fig.clear()
            fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.09,
                                hspace=0.55, wspace=0.4)
            gs_ph = fig.add_gridspec(3, 6,
                                     width_ratios=[1, 1, 1, 1, 1, 1],
                                     height_ratios=[1.1, 0.85, 1.05])

            def _style_ph(ax, ttl):
                ax.set_facecolor(PANEL_BG)
                ax.set_title(ttl, color=TEXT_DIM, fontsize=8, pad=2)
                ax.tick_params(colors=TEXT_COL, labelsize=6)
                for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

            # (slice, title, xlabel, ylabel, is_wide)
            # is_wide: whether this panel is wide enough to show the full
            # centred "Press Start Analysis" hint. Narrow (1-col) panels
            # stay empty so their proportions match the post-click layout
            # exactly.
            panels = [
                (gs_ph[0, 0:2], 'Eigenvalue spectrum',              'Mode index', 'λ (J)',        True),
                (gs_ph[0, 2:4], 'Eigenvec on lattice',              None,         None,            True),
                (gs_ph[0, 4:6], 'Photonic flow',                    None,         None,            True),
                (gs_ph[1, 0:3], 'Mode power vs slow time  (top 5)', 'Iteration',  'Norm. |A|²',    True),
                (gs_ph[1, 3:6], 'Mode power at snapshot',           'Mode index k', 'Norm. |A|²',  True),
                (gs_ph[2, 0:2], '|A(θ,t)|²  (linear)',              'Iteration',  'θ',             True),
                (gs_ph[2, 2:3], '|A(θ)|²  snapshot',                'θ',          None,            False),
                (gs_ph[2, 3:5], '|A(μ,t)|²  (dB)',                  'Iteration',  'FSR μ',         True),
                (gs_ph[2, 5:6], '|A(μ)|²  snapshot',                'FSR μ',      'dB',            False),
            ]
            for spec, ttl, xlab, ylab, is_wide in panels:
                ax = fig.add_subplot(spec)
                _style_ph(ax, ttl)
                if xlab: ax.set_xlabel(xlab, color=TEXT_DIM, fontsize=7)
                if ylab: ax.set_ylabel(ylab, color=TEXT_DIM, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                ax.grid(True, color=GRID_COL, lw=0.4)
                if is_wide:
                    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                            ha='center', va='center', color='#3a4560',
                            fontsize=9, fontstyle='italic')
            canvas.draw()

        _draw_placeholder()

        # ── Control bar: supermode selector | timestep | buttons ───────
        ctrl_row = QHBoxLayout()

        lbl_mode = QLabel('Supermode k:')
        lbl_mode.setStyleSheet('color:#c8d0e7;font-size:12px;')
        spn_mode = QSpinBox(); spn_mode.setRange(0, 0); spn_mode.setValue(0)
        spn_mode.setFixedWidth(60); spn_mode.setEnabled(False)
        sld_mode = QSlider(Qt.Horizontal); sld_mode.setRange(0, 0); sld_mode.setValue(0)
        sld_mode.setFixedWidth(120); sld_mode.setEnabled(False)

        lbl_step = QLabel('Snapshot step:')
        lbl_step.setStyleSheet('color:#c8d0e7;font-size:12px;')
        spn_step = QSpinBox(); spn_step.setRange(0, 0); spn_step.setValue(0)
        spn_step.setFixedWidth(75); spn_step.setEnabled(False)
        sld_step = QSlider(Qt.Horizontal); sld_step.setRange(0, 0); sld_step.setValue(0)
        sld_step.setFixedWidth(160); sld_step.setEnabled(False)

        # keep spinbox and slider in sync
        def _sync_mode_spn(val):
            sld_mode.blockSignals(True); sld_mode.setValue(val); sld_mode.blockSignals(False)
        def _sync_mode_sld(val):
            spn_mode.blockSignals(True); spn_mode.setValue(val); spn_mode.blockSignals(False)
        def _sync_step_spn(val):
            sld_step.blockSignals(True); sld_step.setValue(val); sld_step.blockSignals(False)
        def _sync_step_sld(val):
            spn_step.blockSignals(True); spn_step.setValue(val); spn_step.blockSignals(False)

        spn_mode.valueChanged.connect(_sync_mode_spn)
        sld_mode.valueChanged.connect(_sync_mode_sld)
        spn_step.valueChanged.connect(_sync_step_spn)
        sld_step.valueChanged.connect(_sync_step_sld)

        ctrl_row.addWidget(lbl_mode); ctrl_row.addWidget(spn_mode); ctrl_row.addWidget(sld_mode)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(lbl_step); ctrl_row.addWidget(spn_step); ctrl_row.addWidget(sld_step)
        ctrl_row.addStretch()

        btn_start = QPushButton('▶  Start Analysis')
        btn_start.setStyleSheet('color:#cc88ff;border-color:#cc88ff;')
        btn_stop  = QPushButton('■  Stop'); btn_stop.setEnabled(False)
        btn_save  = QPushButton('💾  Save'); btn_save.setEnabled(False)
        btn_save.setStyleSheet('QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
                               'QPushButton:disabled{color:#2a3050;}')
        # Movie export buttons — time-evolution of k-fixed mode-power bar + A(θ)/A(μ) snaps
        btn_gif = QPushButton('🎞  GIF'); btn_gif.setEnabled(False)
        btn_mp4 = QPushButton('🎬  MP4'); btn_mp4.setEnabled(False)
        _movie_style = ('QPushButton:enabled{color:#ffb84a;border-color:#ffb84a;}'
                        'QPushButton:disabled{color:#2a3050;}')
        btn_gif.setStyleSheet(_movie_style); btn_mp4.setStyleSheet(_movie_style)
        btn_gif.setToolTip('Export time-evolution movie: mode-power bar + |A(θ)|² + |A(μ)|²\n'
                           'for the currently-selected supermode k over a range of snapshot steps.')
        btn_mp4.setToolTip(btn_gif.toolTip())
        btn_close = QPushButton('✕  Close')
        btn_close.setStyleSheet('color:#ff4a6e;border-color:#ff4a6e;')
        for b in (btn_start, btn_stop, btn_save, btn_gif, btn_mp4, btn_close):
            ctrl_row.addWidget(b)
        vl.addLayout(ctrl_row)
        btn_close.clicked.connect(win.close)

        win._stop_flag = threading.Event()
        win._result    = {}
        win._axes      = {}   # store axes refs for interactive updates

        # ── Helper: draw eigenvec lattice ──────────────────────────────
        def _draw_eigenvec(ax, eigvals, eigvecs, k_idx, ISite, OSite, coords, NL):
            ax.cla(); ax.set_facecolor('black'); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            lam_k = eigvals[k_idx]
            ax.set_title(f'|eigvec_{k_idx}|²  on lattice  (λ={lam_k:.4f})',
                         color='#cc88ff', fontsize=8, pad=2)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            mode_vec = np.abs(eigvecs[:, k_idx])**2
            vmax_m   = mode_vec.max() or 1.
            # Bonds (thin, transparent)
            H_mat = p.get('H_mat')
            if H_mat is not None:
                for n in range(1, NL+1):
                    xn, yn = coords[n]
                    for n2 in range(n+1, NL+1):
                        if abs(H_mat[n-1, n2-1]) > 1e-10:
                            x2, y2 = coords[n2]
                            ax.plot([xn,x2],[yn,y2], color='#1e2a40', lw=0.6, zorder=1)
            for n in range(1, NL+1):
                xn, yn = coords[n]
                intensity = float(np.clip(mode_vec[n-1] / vmax_m, 0, 1))
                # Hot colormap for fill, with low alpha so lattice structure shows
                rgba = list(plt.cm.hot(intensity)); rgba[3] = 0.55 + 0.45*intensity
                ec = ('#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else '#556688')
                lw = 2.0 if n in (ISite, OSite) else 1.0
                ax.add_patch(mpatches.Circle((xn,yn), RING_R,
                             edgecolor=ec, facecolor=tuple(rgba), lw=lw, zorder=2))
                ax.add_patch(mpatches.Circle((xn,yn), RING_R*0.62,
                             edgecolor='none', facecolor=(0,0,0,0.7), zorder=3))
            all_x = [v[0] for v in coords.values()]
            all_y = [v[1] for v in coords.values()]
            pad = RING_R * 2
            ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            ax.set_ylim(min(all_y)-pad, max(all_y)+pad)

        K_COLOR = '#ff9500'   # consistent orange for selected supermode k across all panels

        def _update_eigval_highlight(k_idx):
            """Re-draw just the k highlight dot on the eigenvalue spectrum."""
            r = win._result
            if not r: return
            ax = win._axes.get('eigval')
            if ax is None: return
            eigvals = r['eigvals']
            # Remove old k-highlight if present
            for artist in list(ax.collections):
                if getattr(artist, '_modal_k_marker', False):
                    artist.remove()
            sc = ax.scatter([k_idx], [eigvals[k_idx]], color=K_COLOR,
                            s=55, zorder=4, marker='D')
            sc._modal_k_marker = True
            canvas.draw_idle()

        def _update_mode_power(k_idx):
            """Redraw mode-power plot, highlighting mode k in K_COLOR."""
            r = win._result
            if not r: return
            ax = win._axes.get('mode_power')
            if ax is None: return
            eigvals  = r['eigvals']
            top5     = r['top5']
            x_iters  = r['x_iters']
            mode_power_t = r['mode_power_t']
            COLS5 = ['#ffffff','#ffff00','#00ffcc','#ff44cc','#88ff44']
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.set_title('Mode power vs slow time  (top 5)', color=TEXT_COL, fontsize=8, pad=2)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            # Draw top-5 first (thin, background)
            for ci, ki in enumerate(top5):
                if ki == k_idx: continue   # draw k_idx last on top
                lbl = f'λ_{ki}={eigvals[ki]:.3f}'
                if ki == r['closest_idx']: lbl += ' ←Δ'
                ax.plot(x_iters, mode_power_t[ki]/(mode_power_t.max() or 1.),
                        color=COLS5[ci % len(COLS5)], lw=0.9, alpha=0.7, label=lbl)
            # Draw selected k on top in K_COLOR
            lbl_k = f'λ_{k_idx}={eigvals[k_idx]:.3f}  ← k'
            ax.plot(x_iters, mode_power_t[k_idx]/(mode_power_t.max() or 1.),
                    color=K_COLOR, lw=2.0, label=lbl_k, zorder=5)
            ax.legend(fontsize=6, frameon=False, labelcolor=TEXT_COL)
            ax.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel('Norm. |A|²', color=TEXT_DIM, fontsize=7)
            ax.set_xticks([x_iters[0], x_iters[-1]])
            ax.set_xticklabels([f'{int(x_iters[0])}', f'{int(x_iters[-1])}'], fontsize=6)
            ax.grid(True, color=GRID_COL, lw=0.4)
            ax.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel('Norm. |A|²', color=TEXT_DIM, fontsize=7)
            ax.set_xticks([x_iters[0], x_iters[-1]])
            ax.set_xticklabels([f'{int(x_iters[0])}', f'{int(x_iters[-1])}'], fontsize=6)
            ax.grid(True, color=GRID_COL, lw=0.4)

        def _update_flow_panel(k_idx, step_i):
            """Draw 2D rings + photon flow for supermode k at snapshot step_i."""
            r = win._result
            if not r: return
            ax = win._axes.get('flow')
            if ax is None: return
            ax.cla(); ax.set_facecolor('black'); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'Photonic flow  mode {k_idx}  step {step_i}',
                         color=K_COLOR, fontsize=8, pad=2)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

            h_str = ls.get('h_str','IQH_IQH')
            nx0 = ls.get('nx0',4); ny0 = ls.get('ny0',4)
            nx1 = ls.get('nx1',1); ny1 = ls.get('ny1',1)
            NL_real = p['NLattice']
            coords  = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL_real+1)}
            ISite = p['ISite']; OSite = p['OSite']

            # Reconstruct supermode field at each ring (pump FSR mode)
            eigvecs_r = r['eigvecs']          # (NL_real, NL_real)
            NF = r['NF']
            pump_idx  = 0                                    # FFT convention: pump at index 0
            A_mu_snap = r['A_modes_mu'][k_idx, :, step_i]   # (NF,)
            A_pump    = A_mu_snap[pump_idx]                  # scalar amplitude at pump
            # field_1d[m] = eigvecs[m,k] * A_pump  (supermode reconstruction at pump freq)
            field_site = eigvecs_r[:, k_idx] * A_pump        # (NL_real,)
            # Prepend 0 for 1-indexed access
            field_1d = np.zeros(NL_real+1, dtype=complex)
            field_1d[1:] = field_site

            H_mat_full = np.zeros((NL_real+1, NL_real+1), dtype=complex)
            H_mat_full[1:, 1:] = p['H_mat']

            # Draw thin ring outlines only (no fill heatmap)
            for n in range(1, NL_real+1):
                xn, yn = coords[n]
                ec = ('#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else '#2a3a5a')
                lw = 1.5 if n in (ISite, OSite) else 0.6
                ax.add_patch(mpatches.Circle((xn,yn), RING_R,
                             edgecolor=ec, facecolor='none', lw=lw, zorder=2))

            # Photon flow arrows
            Xq, Yq, U, V, cf = compute_flow(
                field_1d, H_mat_full, NL_real, ISite, OSite,
                h_str, nx0, ny0, nx1, ny1)
            if len(Xq) and np.sqrt(U**2 + V**2).max() > 0:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    ax.quiver(Xq, Yq, U, V, color=cf,
                              scale=0.6, scale_units='xy', width=0.012,
                              pivot='middle', zorder=4)

            all_x = [v[0] for v in coords.values()]
            all_y = [v[1] for v in coords.values()]
            is_cyl = ls.get('is_cyl', False)
            is_zz  = h_str == 'A_zigzag'
            if is_cyl or is_zz:
                pad = RING_R * 2
                ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
                ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
            else:
                ax.set_xlim(min(all_x)-0.8, max(all_x)+0.8)
                ax.set_ylim(min(all_y)-0.8, max(all_y)+0.8)

        def _update_mode_bar(k_idx, step_i):
            """Bar chart: mode power at snapshot step, k highlighted in K_COLOR."""
            r = win._result
            if not r: return
            ax = win._axes.get('mode_bar')
            if ax is None: return
            ax.cla(); ax.set_facecolor(PANEL_BG)
            mode_power_t = r['mode_power_t']   # (NL, NS)
            NL_k = mode_power_t.shape[0]
            pbar = mode_power_t[:, step_i]
            pmax = pbar.max() or 1.
            colors = [K_COLOR if i == k_idx else '#2a4a6a' for i in range(NL_k)]
            ax.bar(np.arange(NL_k), pbar / pmax, color=colors, width=0.85)
            ax.set_title(f'Mode power at step {step_i}', color=K_COLOR, fontsize=8, pad=2)
            ax.set_xlabel('Mode index k', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel('Norm. |A|²', color=TEXT_DIM, fontsize=7)
            ax.set_xlim(-0.5, NL_k-0.5)
            ax.set_ylim(0, 1.05)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            ax.set_xticks([0, NL_k//2, NL_k-1])
            ax.set_xticklabels(['0', f'{NL_k//2}', f'{NL_k-1}'], fontsize=6)
            ax.grid(True, color=GRID_COL, lw=0.4, axis='y')
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

        def _update_snapshots(step_i):
            r = win._result
            if not r: return
            A_th_k  = r['A_modes_th'][r['cur_k'], :, :]   # (NF, NS) θ domain
            A_fsr_k = r['A_modes_mu'][r['cur_k'], :, :]   # (NF, NS) μ domain, FFT-order
            NF = r['NF']
            fsr = NF//2
            mu_ax = np.arange(-fsr, fsr+1)[:NF]           # centered μ axis for display

            ax_th = win._axes['snap_th']
            ax_th.cla(); ax_th.set_facecolor(PANEL_BG)
            theta_ax = np.linspace(0, 2*np.pi, NF, endpoint=False)
            ax_th.plot(theta_ax, np.abs(A_th_k[:, step_i])**2,
                       color=K_COLOR, lw=1.2)
            ax_th.set_title(f'|A(θ)|²  step {step_i}', color=K_COLOR, fontsize=8, pad=2)
            ax_th.set_xlabel('θ', color=TEXT_DIM, fontsize=7)
            ax_th.set_xticks([0, np.pi, 2*np.pi])
            ax_th.set_xticklabels(['0','π','2π'], fontsize=6)
            ax_th.tick_params(colors=TEXT_COL, labelsize=6)
            ax_th.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax_th.spines.values(): sp.set_edgecolor('#3a4560')

            ax_mu = win._axes['snap_mu']
            ax_mu.cla(); ax_mu.set_facecolor(PANEL_BG)
            # fftshift along μ axis for centered display
            A_fsr_snap = np.fft.fftshift(A_fsr_k[:, step_i])
            ax_mu.plot(mu_ax, 10*np.log10(np.abs(A_fsr_snap)**2 + 1e-20),
                       color=K_COLOR, lw=1.2)
            ax_mu.set_title(f'|A(μ)|²  step {step_i}  (dB)', color=K_COLOR, fontsize=8, pad=2)
            ax_mu.set_xlabel('FSR μ', color=TEXT_DIM, fontsize=7)
            ax_mu.set_ylabel('dB', color=TEXT_DIM, fontsize=7)
            ax_mu.set_xticks([-fsr, 0, fsr])
            ax_mu.set_xticklabels([f'{-fsr}','0',f'{fsr}'], fontsize=6)
            ax_mu.tick_params(colors=TEXT_COL, labelsize=6)
            ax_mu.grid(True, color=GRID_COL, lw=0.4)
            for sp in ax_mu.spines.values(): sp.set_edgecolor('#3a4560')

            # Move vlines
            x_val = r['x_iters'][step_i]
            for vl_attr in ('vline_mu', 'vline_th'):
                vl_line = win._axes.get(vl_attr)
                if vl_line:
                    vl_line.set_xdata([x_val, x_val])
            _update_flow_panel(r['cur_k'], step_i)
            _update_mode_bar(r['cur_k'], step_i)
            canvas.draw_idle()

        def _update_supermode(k_idx):
            r = win._result
            if not r: return
            r['cur_k'] = k_idx
            eigvals = r['eigvals']; eigvecs = r['eigvecs']
            NL = r['NL']; NF = r['NF']; NS = r['NS']
            x_iters = r['x_iters']
            ISite = p['ISite']; OSite = p['OSite']
            h_str = ls.get('h_str','IQH_IQH')
            nx0 = ls.get('nx0',4); ny0 = ls.get('ny0',4)
            nx1 = ls.get('nx1',1); ny1 = ls.get('ny1',1)
            coords = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}

            # Redraw eigenvec lattice
            _draw_eigenvec(win._axes['eigvec'], eigvals, eigvecs,
                           k_idx, ISite, OSite, coords, NL)

            # Highlight k on eigenvalue spectrum and mode power plot
            _update_eigval_highlight(k_idx)
            _update_mode_power(k_idx)

            # Redraw heatmaps for new k
            A_th_k  = r['A_modes_th'][k_idx, :, :]   # (NF, NS) — θ domain
            A_fsr_k = r['A_modes_mu'][k_idx, :, :]   # (NF, NS) — μ domain
            # A_fsr_k is in FFT order; power_th_k has no shift (θ axis native).
            power_th_k = np.abs(A_th_k)**2
            power_mu_k = np.fft.fftshift(np.abs(A_fsr_k)**2, axes=0)   # centered μ for display
            fsr = NF//2

            ax_hth = win._axes['hmap_th']
            ax_hth.cla(); ax_hth.set_facecolor(PANEL_BG)
            ax_hth.imshow(power_th_k, aspect='auto',
                          origin='lower', cmap='viridis',
                          extent=[x_iters[0], x_iters[-1], 0, 2*np.pi],
                          interpolation='nearest')
            ax_hth.set_title(f'|A(θ,t)|²  mode {k_idx}', color=K_COLOR, fontsize=8, pad=2)
            ax_hth.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax_hth.set_ylabel('θ', color=TEXT_DIM, fontsize=7)
            ax_hth.set_yticks([0, np.pi, 2*np.pi])
            ax_hth.set_yticklabels(['0','π','2π'], fontsize=6)
            ax_hth.set_xticks([x_iters[0], x_iters[-1]])
            ax_hth.set_xticklabels([f'{int(x_iters[0])}', f'{int(x_iters[-1])}'], fontsize=6)
            ax_hth.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax_hth.spines.values(): sp.set_edgecolor('#3a4560')
            win._axes['vline_th'] = ax_hth.axvline(
                x=x_iters[spn_step.value()], color='white', lw=1.0, ls=':', alpha=0.9)

            ax_hmu = win._axes['hmap_mu']
            ax_hmu.cla(); ax_hmu.set_facecolor(PANEL_BG)
            ax_hmu.imshow(10*np.log10(power_mu_k + 1e-20), aspect='auto',
                          origin='lower', cmap='hot',
                          extent=[x_iters[0], x_iters[-1], -fsr, fsr],
                          interpolation='nearest')
            ax_hmu.set_title(f'|A(μ,t)|²  mode {k_idx}  (dB)', color=K_COLOR, fontsize=8, pad=2)
            ax_hmu.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax_hmu.set_ylabel('FSR μ', color=TEXT_DIM, fontsize=7)
            ax_hmu.set_yticks([-fsr, 0, fsr]); ax_hmu.set_yticklabels([f'{-fsr}','0',f'{fsr}'], fontsize=6)
            ax_hmu.set_xticks([x_iters[0], x_iters[-1]])
            ax_hmu.set_xticklabels([f'{int(x_iters[0])}', f'{int(x_iters[-1])}'], fontsize=6)
            ax_hmu.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax_hmu.spines.values(): sp.set_edgecolor('#3a4560')
            win._axes['vline_mu'] = ax_hmu.axvline(
                x=x_iters[spn_step.value()], color='white', lw=1.0, ls=':', alpha=0.9)

            _update_snapshots(spn_step.value())

        def _run_analysis():
            btn_start.setEnabled(False); btn_stop.setEnabled(True)
            _draw_placeholder('Analysis ongoing…')
            win._stop_flag.clear()
            self._modal_status.setText('Computing eigenmodes…')
            QApplication.processEvents()

            psi_val = p.get('psi', 0.0)
            if ls.get('is_cyl') and build_hamiltonian is not None:
                H_full = build_hamiltonian(
                    ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                    ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2),
                    ls.get('phi_iqh1', np.pi/2), ls.get('phi_aqh0', np.pi/4),
                    ls.get('phi_aqh1', np.pi/4), psi_val)
                H = H_full[1:, 1:].copy()
            else:
                H = p['H_mat'].copy()

            try:
                eigvals, eigvecs = np.linalg.eigh(H)
            except Exception as e:
                self._modal_status.setText(f'Eigensolve failed: {e}')
                btn_start.setEnabled(True); btn_stop.setEnabled(False); return

            detuning    = p['detuning']
            closest_idx = int(np.argmin(np.abs(eigvals - detuning)))
            closest_val = eigvals[closest_idx]
            NL, NF, NS  = self._hist.shape
            every       = 1   # save_every is hardcoded to 1 (no more spinbox)
            # Global step axis: offset by total steps completed in earlier rounds.
            if self._round_niters and NS > 0:
                steps_before_hist = sum(self._round_niters[:-1])
            else:
                steps_before_hist = 0
            x_iters     = steps_before_hist + np.arange(NS) * every

            self._modal_status.setText('Projecting field onto eigenmodes…')
            QApplication.processEvents()
            if win._stop_flag.is_set():
                btn_start.setEnabled(True); btn_stop.setEnabled(False); return

            # A(k, θ, t) = Σ_m v*_mk · a_T(m, θ, t)  — project directly in θ domain
            # hist[m, θ, t] is already in fast-time (θ) domain
            A_modes_th = np.einsum('mk,mft->kft', eigvecs.conj(), self._hist)  # (NL,NF,NS)
            # A(k, μ, t) = FFT of A(k,θ,t) over θ axis
            A_modes_mu = np.fft.fft(A_modes_th, axis=1, norm='ortho')          # (NL,NF,NS)

            if win._stop_flag.is_set():
                btn_start.setEnabled(True); btn_stop.setEnabled(False); return

            mode_power_t  = np.sum(np.abs(A_modes_th)**2, axis=1)   # (NL, NS) — power in θ
            total_mp      = mode_power_t.sum(axis=1)
            top5          = list(np.argsort(total_mp)[::-1][:5])

            win._result = dict(
                eigvals=eigvals, eigvecs=eigvecs,
                closest_idx=closest_idx, closest_val=closest_val,
                cur_k=closest_idx,
                A_modes_th=A_modes_th, A_modes_mu=A_modes_mu,
                mode_power_t=mode_power_t, top5=top5,
                x_iters=x_iters, NL=NL, NF=NF, NS=NS, detuning=detuning)

            self._modal_status.setText('Building layout…')
            QApplication.processEvents()

            # ── Build static layout ────────────────────────────────────
            fig.clear()
            fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.09,
                                hspace=0.55, wspace=0.4)
            gs = fig.add_gridspec(3, 6,
                                  width_ratios=[1, 1, 1, 1, 1, 1],
                                  height_ratios=[1.1, 0.85, 1.05])

            def _style(ax, ttl, col):
                ax.set_facecolor(PANEL_BG)
                ax.set_title(ttl, color=col, fontsize=8, pad=2)
                ax.tick_params(colors=TEXT_COL, labelsize=6)
                for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

            h_str = ls.get('h_str','IQH_IQH')
            nx0 = ls.get('nx0',4); ny0 = ls.get('ny0',4)
            nx1 = ls.get('nx1',1); ny1 = ls.get('ny1',1)
            coords = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}
            ISite = p['ISite']; OSite = p['OSite']
            fsr = NF//2

            # ── Row 0: Eigenvalue spectrum | Eigenvec | Photonic flow ──
            ax1 = fig.add_subplot(gs[0, 0:2])
            _style(ax1, 'Eigenvalue spectrum', TEXT_COL)
            ax1.scatter(np.arange(NL), eigvals, color='#c8d0e7', s=8, zorder=2)
            ax1.scatter([closest_idx], [closest_val], color='#ff4a6e', s=40, zorder=3)
            ax1.axhline(detuning, color=ACCENT, lw=1.0, ls='--', alpha=0.8)
            ax1.set_xlabel('Mode index', color=TEXT_DIM, fontsize=7)
            ax1.set_ylabel('λ (J)', color=TEXT_DIM, fontsize=7)
            ax1.grid(True, color=GRID_COL, lw=0.4)
            ax1.set_xticks([0, NL-1]); ax1.set_xticklabels(['0', f'{NL-1}'], fontsize=6)
            win._axes['eigval'] = ax1

            ax2 = fig.add_subplot(gs[0, 2:4])
            win._axes['eigvec'] = ax2
            _draw_eigenvec(ax2, eigvals, eigvecs, closest_idx, ISite, OSite, coords, NL)

            ax_flow = fig.add_subplot(gs[0, 4:6])
            ax_flow.set_facecolor('black'); ax_flow.set_aspect('equal')
            ax_flow.set_xticks([]); ax_flow.set_yticks([])
            win._axes['flow'] = ax_flow

            # ── Row 1: Mode power vs time | Mode power bar ────────────
            ax3 = fig.add_subplot(gs[1, 0:3])
            _style(ax3, 'Mode power vs slow time  (top 5)', TEXT_COL)
            ax3.grid(True, color=GRID_COL, lw=0.4)
            win._axes['mode_power'] = ax3

            ax_bar = fig.add_subplot(gs[1, 3:6])
            win._axes['mode_bar'] = ax_bar

            # ── Row 2: A(θ,t) | snap_θ | A(μ,t) | snap_μ ─────────────
            ax4 = fig.add_subplot(gs[2, 0:2]); win._axes['hmap_th'] = ax4
            ax5 = fig.add_subplot(gs[2, 2:3]); win._axes['snap_th'] = ax5
            ax6 = fig.add_subplot(gs[2, 3:5]); win._axes['hmap_mu'] = ax6
            ax7 = fig.add_subplot(gs[2, 5:6]); win._axes['snap_mu'] = ax7

            # Enable controls
            spn_mode.setRange(0, NL-1); spn_mode.setValue(closest_idx)
            sld_mode.setRange(0, NL-1); sld_mode.setValue(closest_idx)
            spn_mode.setEnabled(True);  sld_mode.setEnabled(True)
            spn_step.setRange(0, NS-1); spn_step.setValue(NS-1)
            sld_step.setRange(0, NS-1); sld_step.setValue(NS-1)
            spn_step.setEnabled(True);  sld_step.setEnabled(True)

            btn_stop.setEnabled(False); btn_save.setEnabled(True)
            btn_gif.setEnabled(True); btn_mp4.setEnabled(True)

            # Draw heatmaps + snapshots for closest mode
            _update_supermode(closest_idx)

            self._modal_status.setText(
                f'Done  |  closest: λ_{closest_idx}={closest_val:.5f}  '
                f'(Δλ={abs(closest_val-detuning):.5f})  |  '
                f'Use "Supermode k" to switch mode, "Snapshot step" to move vline.')

        def _on_step_changed(val):
            _update_snapshots(val)
            canvas.draw_idle()

        def _on_mode_changed(val):
            _update_supermode(val)

        def _save():
            folder = p.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
            os.makedirs(folder, exist_ok=True)
            for fmt, dpi in [('png', 200), ('svg', 150)]:
                buf = io.BytesIO()
                fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                with open(os.path.join(folder, f'modal_decomposition.{fmt}'), 'wb') as f:
                    f.write(buf.getvalue())
            r = win._result
            if r:
                np.savez(os.path.join(folder, 'modal_decomposition.npz'),
                         eigvals=r['eigvals'],
                         mode_power_t=r['mode_power_t'],
                         x_iters=r['x_iters'])
            self._modal_status.setText(f'✅ Saved → {os.path.basename(folder)}')

        # ── Movie export: time-evolution of mode-power bar + A(θ) + A(μ) ──
        # Sweeps the snapshot step axis with the currently-selected supermode k
        # fixed. Three stacked panels per frame. Uses global (over-range) y-axis
        # maxima so bars and snapshot curves are comparable across frames.
        def _export_movie(fmt):
            r = win._result
            if not r:
                self._modal_status.setText('Run analysis first.'); return
            NS = r['NS']
            k_fixed = r['cur_k']

            # ── Dialog for frame range / interval / fps ──────────────
            dlg = QDialog(win)
            dlg.setWindowTitle(f'Export {fmt.upper()} — modal time evolution')
            dlg.setStyleSheet(STYLE); dlg.setMinimumWidth(340)
            fl = QGridLayout(dlg); fl.setSpacing(6)
            def _lb(t):
                lb = QLabel(t); lb.setStyleSheet('color:#c8d0e7;font-size:13px;'); return lb

            spn_s = QSpinBox(); spn_s.setRange(0, NS-1); spn_s.setValue(max(0, NS-1001))
            spn_e = QSpinBox(); spn_e.setRange(0, NS-1); spn_e.setValue(NS-1)
            spn_t = QSpinBox(); spn_t.setRange(1, max(1, NS//2)); spn_t.setValue(10)
            spn_t.setToolTip('Save every N-th snapshot as a frame')
            spn_fps = QSpinBox(); spn_fps.setRange(1, 60)
            spn_fps.setValue(5 if fmt == 'gif' else 10)
            spn_fps.setToolTip('Playback frame rate (frames per second)')

            fl.addWidget(_lb(f'Fixed supermode k = {k_fixed}'), 0, 0, 1, 2)
            fl.addWidget(_lb('Start step:'),    1, 0); fl.addWidget(spn_s,   1, 1)
            fl.addWidget(_lb('End step:'),      2, 0); fl.addWidget(spn_e,   2, 1)
            fl.addWidget(_lb('Step interval:'), 3, 0); fl.addWidget(spn_t,   3, 1)
            fl.addWidget(_lb('Playback FPS:'),  4, 0); fl.addWidget(spn_fps, 4, 1)

            lbl_n = QLabel(''); lbl_n.setStyleSheet('color:#4a5270;font-size:12px;')
            fl.addWidget(lbl_n, 5, 0, 1, 2)
            def _update_n(*_):
                s = spn_s.value(); e = spn_e.value(); t = spn_t.value(); f_ = spn_fps.value()
                n = max(0, (e - s) // t + 1) if e >= s else 0
                dur = n / f_ if f_ > 0 else 0
                lbl_n.setText(f'→  {n} frames  |  duration ≈ {dur:.1f} s at {f_} fps')
            for w in (spn_s, spn_e, spn_t, spn_fps): w.valueChanged.connect(_update_n)
            _update_n()

            btn_row = QHBoxLayout()
            btn_ok  = QPushButton('Export'); btn_ok.clicked.connect(dlg.accept)
            btn_can = QPushButton('Cancel'); btn_can.clicked.connect(dlg.reject)
            btn_row.addStretch(); btn_row.addWidget(btn_ok); btn_row.addWidget(btn_can)
            fl.addLayout(btn_row, 6, 0, 1, 2)

            if dlg.exec_() != QDialog.Accepted: return

            s_start = spn_s.value(); s_end = spn_e.value(); s_step = spn_t.value()
            fps_val = spn_fps.value()
            if s_end < s_start: s_start, s_end = s_end, s_start
            steps = list(range(s_start, s_end + 1, s_step))
            if not steps:
                self._modal_status.setText('No frames in range.'); return

            folder = p.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
            os.makedirs(folder, exist_ok=True)
            fname = (f'modal_movie_k{k_fixed}_s{s_start}_e{s_end}'
                     f'_step{s_step}_fps{fps_val}.{fmt}')
            out   = os.path.join(folder, fname)

            # ── Pre-compute global y-limits so frames are comparable ──
            mode_power_t = r['mode_power_t']          # (NL, NS)
            A_th_k       = r['A_modes_th'][k_fixed]    # (NF, NS) θ domain
            A_mu_k       = r['A_modes_mu'][k_fixed]    # (NF, NS) μ domain, FFT-order
            NF           = r['NF']
            NL_bar       = mode_power_t.shape[0]

            # For bars: global max across the ENTIRE movie range, not just per-frame,
            # so growing/shrinking power is visible.
            bar_slice    = mode_power_t[:, steps]
            bar_gmax     = float(bar_slice.max() or 1.)

            pwr_th_k     = np.abs(A_th_k[:, steps])**2
            pwr_th_gmax  = float(pwr_th_k.max() or 1.)

            pwr_mu_k     = np.abs(A_mu_k[:, steps])**2
            # Centered μ for display; same fftshift as _update_snapshots
            pwr_mu_shift = np.fft.fftshift(pwr_mu_k, axes=0)
            # dB floor: use 1e-20 to match on-screen snapshot plot
            pwr_mu_db    = 10*np.log10(pwr_mu_shift + 1e-20)
            mu_dB_top    = float(np.ceil(pwr_mu_db.max() + 1.0))
            mu_dB_bot    = float(np.floor(pwr_mu_db.min() - 1.0))

            fsr      = NF // 2
            mu_ax    = np.arange(-fsr, fsr + 1)[:NF]
            theta_ax = np.linspace(0, 2*np.pi, NF, endpoint=False)
            eigvals  = r['eigvals']
            x_iters  = r['x_iters']
            lam_k    = eigvals[k_fixed]

            self._modal_status.setText(f'Exporting {len(steps)} frames…')
            QApplication.processEvents()

            # ── Build export figure: 3 stacked panels ─────────────────
            fig_exp = plt.figure(figsize=(7, 8), facecolor=DARK_BG)
            fig_exp.subplots_adjust(left=0.11, right=0.96, top=0.94, bottom=0.07,
                                    hspace=0.55)
            ax_bar_e = fig_exp.add_subplot(3, 1, 1)
            ax_th_e  = fig_exp.add_subplot(3, 1, 2)
            ax_mu_e  = fig_exp.add_subplot(3, 1, 3)

            def _style_exp(ax):
                ax.set_facecolor(PANEL_BG)
                ax.tick_params(colors=TEXT_COL, labelsize=7)
                ax.grid(True, color=GRID_COL, lw=0.4)
                for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

            def _frame(si):
                for ax in (ax_bar_e, ax_th_e, ax_mu_e): ax.cla()

                # ── Top: mode-power bar chart (k highlighted) ─────
                _style_exp(ax_bar_e)
                pbar = mode_power_t[:, si]
                colors = [K_COLOR if i == k_fixed else '#2a4a6a' for i in range(NL_bar)]
                ax_bar_e.bar(np.arange(NL_bar), pbar / bar_gmax,
                             color=colors, width=0.85)
                ax_bar_e.set_title(
                    f'Mode power   step {si}   (k = {k_fixed}, λ = {lam_k:.4f})',
                    color=K_COLOR, fontsize=10, pad=4)
                ax_bar_e.set_xlabel('Mode index k', color=TEXT_DIM, fontsize=9)
                ax_bar_e.set_ylabel('Norm. |A|²  (global)', color=TEXT_DIM, fontsize=9)
                ax_bar_e.set_xlim(-0.5, NL_bar - 0.5)
                ax_bar_e.set_ylim(0, 1.05)

                # ── Mid: |A(θ)|²  for this k at step si ───────────
                _style_exp(ax_th_e)
                ax_th_e.plot(theta_ax, np.abs(A_th_k[:, si])**2,
                             color=K_COLOR, lw=1.4)
                ax_th_e.set_title(f'|A(θ)|²   step {si}', color=K_COLOR,
                                  fontsize=10, pad=4)
                ax_th_e.set_xlabel('θ', color=TEXT_DIM, fontsize=9)
                ax_th_e.set_xticks([0, np.pi, 2*np.pi])
                ax_th_e.set_xticklabels(['0', 'π', '2π'], fontsize=8)
                ax_th_e.set_xlim(0, 2*np.pi)
                ax_th_e.set_ylim(0, pwr_th_gmax * 1.05)

                # ── Bot: |A(μ)|² dB  for this k at step si ────────
                _style_exp(ax_mu_e)
                A_snap_db = 10*np.log10(
                    np.fft.fftshift(np.abs(A_mu_k[:, si])**2) + 1e-20)
                ax_mu_e.plot(mu_ax, A_snap_db, color=K_COLOR, lw=1.4)
                ax_mu_e.set_title(f'|A(μ)|²   step {si}   (dB)',
                                  color=K_COLOR, fontsize=10, pad=4)
                ax_mu_e.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=9)
                ax_mu_e.set_ylabel('dB', color=TEXT_DIM, fontsize=9)
                ax_mu_e.set_xticks([-fsr, 0, fsr])
                ax_mu_e.set_xticklabels([f'{-fsr}', '0', f'{fsr}'], fontsize=8)
                ax_mu_e.set_xlim(-fsr, fsr)
                ax_mu_e.set_ylim(mu_dB_bot, mu_dB_top)
                return []

            try:
                if fmt == 'gif':
                    import matplotlib.animation as animation
                    ani = animation.FuncAnimation(
                        fig_exp, _frame, frames=steps, blit=False)
                    ani.save(out, writer='pillow', fps=fps_val)
                else:
                    try:
                        import imageio
                        from PIL import Image as _PILImage
                    except ImportError:
                        plt.close(fig_exp)
                        self._modal_status.setText(
                            '❌ MP4 needs imageio: pip install imageio imageio-ffmpeg')
                        return
                    frames_arr = []
                    for si in steps:
                        _frame(si)
                        buf = io.BytesIO()
                        fig_exp.savefig(buf, format='png', dpi=100,
                                        facecolor=fig_exp.get_facecolor())
                        buf.seek(0)
                        frames_arr.append(
                            np.array(_PILImage.open(buf).convert('RGB')))
                    imageio.mimwrite(out, frames_arr, fps=fps_val, quality=8)
            finally:
                plt.close(fig_exp)

            self._modal_status.setText(f'✅ Saved → {os.path.basename(out)}')

        btn_gif.clicked.connect(lambda: _export_movie('gif'))
        btn_mp4.clicked.connect(lambda: _export_movie('mp4'))

        spn_step.valueChanged.connect(_on_step_changed)
        sld_step.valueChanged.connect(_on_step_changed)
        spn_mode.valueChanged.connect(_on_mode_changed)
        sld_mode.valueChanged.connect(_on_mode_changed)
        # Run on the main thread — _run_analysis calls QApplication.processEvents()
        # at several points, keeping the UI (including Stop) responsive.
        # A background thread is unsafe here because _run_analysis directly manipulates
        # Qt widgets (setEnabled, setText, canvas.draw) which must only happen on the
        # main thread; doing so from a worker thread causes intermittent crashes.
        btn_start.clicked.connect(_run_analysis)
        btn_stop.clicked.connect(lambda: win._stop_flag.set())
        btn_save.clicked.connect(_save)

        win.show()

    # ------------------------------------------------------------------
    def _draw_all(self):
        # Set interrupt flag for Exit button
        self._plotting = True

        # Collect all interactive widgets to disable
        self._plot_widgets = [
            self.btn_run, self.btn_stop, self.btn_save, self.btn_save_final,
            self.btn_lyapunov, self.btn_modal,
            self.btn_replot, self.btn_vis_plot, self.btn_vis_save,
            self.btn_vis_gif, self.btn_vis_mp4,
            self.btn_vis_prev, self.btn_vis_next,
            self.spn_niter,
            self.spn_fsr_shift, self.spn_mode_idx,
            self.spn_nest_range, self.spn_ycut_freq,
            self.spn_fsr_shift, self.spn_vis_step,
        ]
        for w in self._plot_widgets:
            w.setEnabled(False)
        # Exit always enabled — clicking it sets _plotting=False and closes
        self.btn_vis_exit.setEnabled(True)

        steps_total = 6
        def _progress(step, label):
            if not self._plotting: return   # interrupted by Exit
            pct = int(100 * step / steps_total)
            self.prog.setValue(pct)
            self.status.showMessage(f'Plotting… {label}')
            QApplication.processEvents()

        # Show "Plotting…" on all panels except the static sweep indicator
        plotting_panels = [
            ([self.ax_aT_ring, self.ax_aT_all, self.ax_aW_ring, self.ax_aW_all], self.canvas_heat4),
            ([self.ax_comb_dB, self.ax_comb_zoom],                                self.canvas_comb),
            ([self.ax_vis_2d],                                                     self.canvas_2dpow),
            ([self.ax_esa_dB, self.ax_esa_lin, self.ax_osc],                      self.canvas_esa),
            ([self.ax_hmap, self.ax_ycut_pwr, self.ax_ycut_phs],                  self.canvas_spec2d),
            ([self.ax_vis_flow],                                                   self.canvas_flow),
        ]
        for axes, canvas in plotting_panels:
            for ax in axes:
                ax.cla(); ax.set_facecolor(PANEL_BG)
                ax.text(0.5, 0.5, 'Plotting…', transform=ax.transAxes,
                        ha='center', va='center', color=TEXT_COL, fontsize=11)
            canvas.draw_idle()
        # 3D axes needs x,y,z coords, not transAxes
        self.ax_vis_3d.cla(); self.ax_vis_3d.set_facecolor('black')
        self.ax_vis_3d.text(0.5, 0.5, 0.5, 'Plotting…', ha='center', va='center',
                            color=TEXT_COL, fontsize=11)
        self.canvas_3d.draw_idle()
        QApplication.processEvents()

        hist     = self._hist                       # (NLattice, NumFSRs, NS)
        p        = self._p
        OSite_i  = p['OSite'] - 1
        FSRVec   = p['FSRVec']                      # FFT-ordered [0,1,...,+N,-N,...,-1]
        FSR_s    = np.fft.fftshift(FSRVec)           # centered [-N,...,0,...,+N] — for plotting
        one_side = p['one_side_fsr']
        PumpFSR  = 0                                 # FFT convention: pump at index 0
        NumFSRs  = len(FSRVec)
        NL, NF, NS = hist.shape
        every    = 1                                 # hardcoded — no more spinbox
        dt       = p['TStep'] * every
        # Global step axis — aligns across rounds. For round k, x_iters
        # starts at the total number of steps evolved in rounds 1..k-1 and
        # runs through that + NS. This way a user running 10k steps in round
        # 1 then 2k in round 2 sees the round 2 plot x-axis span 10k..12k.
        # When there's no round history (legacy "run from fresh" path),
        # steps_before_hist is 0 and we match the old behavior.
        if self._round_niters and NS > 0:
            # Frames in self._hist belong to the LAST completed round,
            # which is self._round_niters[-1] steps long. So the first frame
            # of hist was produced at step (sum of previous rounds) + 1.
            steps_before_hist = sum(self._round_niters[:-1])
        else:
            steps_before_hist = 0
        x_iters  = steps_before_hist + np.arange(NS) * every

        shift_factor  = self.spn_fsr_shift.value()
        mode_index    = self.spn_mode_idx.value()   # relative to pump, i.e. physical μ
        nest_range    = self.spn_nest_range.value()
        # Map physical μ → FFT-order array index (handles negatives correctly via modulo)
        index         = mode_index % NumFSRs
        mode_indices  = FSR_s                       # physical μ per centered-display column

        # ── shared data ───────────────────────────────────────────────
        # time-domain (θ axis — no shift)
        aT_ring = np.abs(hist[OSite_i, :, :])**2                    # (NF, NS)
        aT_all  = np.sum(np.abs(hist)**2, axis=0)                   # (NF, NS)

        # frequency-domain (FFT along FSR axis; fftshift for centered display)
        aW_full     = np.fft.fft(hist, axis=1, norm='ortho')        # (NL, NF, NS)
        aW_ring_dB  = 10*np.log10(np.fft.fftshift(np.abs(aW_full[OSite_i, :, :])**2, axes=0) + 1e-200)
        aW_all_dB   = 10*np.log10(np.fft.fftshift(np.sum(np.abs(aW_full)**2, axis=0), axes=0) + 1e-200)

        theta_idx = np.arange(NF)

        # slow-time FFT of each comb tooth's complex amplitude.
        # Shift rows to centered μ order so row i corresponds to mode_indices[i].
        a_W_ring    = np.fft.fftshift(aW_full[OSite_i, :, :], axes=0)   # (NF, NS) — row i = μ=FSR_s[i]
        freq_J      = np.fft.fftshift(np.fft.fftfreq(NS, d=dt)) * 2*np.pi
        A_freq      = np.fft.fftshift(np.fft.fft(a_W_ring, axis=1), axes=1)
        power_slow  = np.abs(A_freq)**2
        power_slow_dB = 10*np.log10(power_slow + 1e-200)

        freq_mask     = (freq_J >= -nest_range) & (freq_J <= nest_range)
        freq_trimmed  = freq_J[freq_mask]
        pwr_dB_trim   = power_slow_dB[:, freq_mask]
        pwr_trim      = power_slow[:, freq_mask]

        # y-cut at freq=0
        freq_idx      = np.argmin(np.abs(freq_trimmed))
        pwr_ycut_dB   = pwr_dB_trim[:, freq_idx]
        A_ycut        = A_freq[:, freq_mask][:, freq_idx]
        phase_ycut    = np.unwrap(np.angle(A_ycut))

        # ESA: FFT of intensity |a_T|² over slow time
        a_T_ring_raw  = hist[OSite_i, :, :]                         # (NF, NS) complex
        I_t_modes     = np.abs(a_T_ring_raw)**2
        freq_esa      = np.fft.rfftfreq(NS, d=dt) * 2*np.pi
        spec_esa      = np.fft.rfft(I_t_modes, axis=1)
        pwr_esa       = np.abs(spec_esa)**2
        total_pwr_esa = np.sum(pwr_esa, axis=0)
        total_pwr_esa_dB = 10*np.log10(total_pwr_esa + 1e-20)

        # oscilloscope: comb power (pump-excluded) vs slow time.
        # Sum |a_W|² across all sideband modes at the output ring, excluding
        # the pump bin (μ=0, which after fftshift sits at index NF//2). The
        # pump is ~constant by construction and, being 20–40 dB above signal,
        # drowns out the breather dynamics if included. Excluding it gives
        # the clean "comb power" trace that standard Kerr-comb papers show.
        I_w_ring = np.abs(a_W_ring)**2                              # (NF, NS)
        pump_bin = NF // 2
        mask     = np.ones(NF, dtype=bool); mask[pump_bin] = False
        pulse    = np.sum(I_w_ring[mask, :], axis=0)
        pulse    = pulse / (pulse.max() or 1.)

        # ── helper ────────────────────────────────────────────────────
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes

        def _pcm(ax, Z, xv, yv, cmap, col, ttl, xl, yl):
            ax.cla(); ax.set_facecolor(PANEL_BG)
            vmax = float(Z.max())
            vmin = vmax - 60 if cmap == 'hot' else 0.
            Zn   = Z if cmap == 'hot' else Z / (Z.max() or 1.)
            dx   = (xv[1]-xv[0]) if len(xv) > 1 else max(every, 1)
            dy   = yv[1]-yv[0] if len(yv) > 1 else 1.
            xe   = np.append(xv, xv[-1]+dx)
            ye   = np.append(yv, yv[-1]+dy)
            mesh = ax.pcolormesh(xe, ye, Zn, cmap=cmap,
                          vmin=vmin if cmap=='hot' else 0,
                          vmax=vmax if cmap=='hot' else 1,
                          shading='flat')
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            # x-axis: leftmost and rightmost only (iteration/step axis)
            ax.set_xticks([xv[0], xv[-1]])
            ax.set_xticklabels([f'{int(xv[0])}', f'{int(xv[-1])}'], fontsize=6)
            # y-axis ticks
            if cmap == 'hot':   # a_W: 3 ticks at -FSR, 0, +FSR
                fsr = one_side
                ax.set_yticks([-fsr, 0, fsr])
                ax.set_yticklabels([f'{-fsr}', '0', f'{fsr}'], fontsize=6)
            else:               # a_T: 2 ticks at 0 and 2π
                ax.set_yticks([yv[0], yv[-1]])
                ax.set_yticklabels(['0', '2π'], fontsize=6)
            return mesh

        def _add_cbar(ax, mesh):
            """Add a slim inset colorbar with only min and max ticks.
            Remove any previously added inset axes first to avoid pileup on re-run."""
            # Remove old inset axes parented to this ax (ax.cla() does not remove them)
            for child in list(ax.child_axes if hasattr(ax, 'child_axes') else []):
                child.remove()
            cax = _inset_axes(ax, width='4%', height='90%',
                              loc='center right',
                              bbox_to_anchor=(0, 0, 1.06, 1),
                              bbox_transform=ax.transAxes,
                              borderpad=0)
            cb = self.fig_heat4.colorbar(mesh, cax=cax)
            clim = mesh.get_clim()
            cb.set_ticks([clim[0], clim[1]])
            cb.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
            cb.ax.tick_params(colors=TEXT_COL, labelsize=5, pad=2)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color(TEXT_COL); lbl.set_fontsize(5); lbl.set_clip_on(False)

        # cache everything needed for live replot (no recalc)
        self._slow_cache = dict(
            freq_J=freq_J, power_slow=power_slow, power_slow_dB=power_slow_dB,
            mode_indices=mode_indices, one_side=one_side,
            pwr_dB_trim=pwr_dB_trim, freq_trimmed=freq_trimmed,
            A_freq=A_freq,
        )

        # ── Block 1: comb dB + zoomed tooth ──────────────────────────
        _progress(1, 'comb spectrum')
        if not self._plotting: return
        self._draw_comb()
        self._draw_zoom()

        # ── Block 2: ESA dB + ESA linear + oscilloscope ───────────────
        _progress(2, 'ESA & oscilloscope')
        if not self._plotting: return
        df = 2*np.pi / (NS * dt)
        self.ax_esa_dB.cla(); self.ax_esa_dB.set_facecolor(PANEL_BG)
        self.ax_esa_dB.plot(freq_esa, total_pwr_esa_dB, color='#a8ff78', lw=1.2)
        self.ax_esa_dB.set_xlim([0, 4]); self.ax_esa_dB.set_xlabel('Angular freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_dB.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_dB.set_title(f'ESA (dB)  Δf≈{df:.3f} J', color='#a8ff78', fontsize=8, pad=2)
        self.ax_esa_dB.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_esa_dB.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_esa_dB.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_esa_lin.cla(); self.ax_esa_lin.set_facecolor(PANEL_BG)
        self.ax_esa_lin.plot(freq_esa, total_pwr_esa/(total_pwr_esa.max() or 1.),
                             color='#a8ff78', lw=1.2)
        self.ax_esa_lin.set_xlim([0, 4]); self.ax_esa_lin.set_ylim([0, 0.5])
        self.ax_esa_lin.set_xlabel('Angular freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_lin.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=7)
        self.ax_esa_lin.set_title('ESA (linear)', color='#a8ff78', fontsize=8, pad=2)
        self.ax_esa_lin.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_esa_lin.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_esa_lin.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_osc.cla(); self.ax_osc.set_facecolor(PANEL_BG)
        self.ax_osc.plot(x_iters, pulse, color='#c8d0e7', lw=1.0)
        # Span only the current round's steps, not 0→end. For round-k data
        # starting at global step `steps_before_hist`, this keeps the trace
        # filling the panel instead of sitting in a small window at the far right.
        self.ax_osc.set_xlim([x_iters[0], max(x_iters[-1], x_iters[0] + 1)])
        self.ax_osc.set_ylim([0, 1.1])
        self.ax_osc.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
        self.ax_osc.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=7)
        self.ax_osc.set_title(f'Comb power vs slow time  (pump-excluded, osite)',
                              color='#c8d0e7', fontsize=8, pad=2)
        self.ax_osc.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_osc.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_osc.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)
        self.canvas_esa.draw_idle()

        # ── Block 3: 2D heatmap + y-cuts ─────────────────────────────
        _progress(3, '2D slow-time spectrum')
        if not self._plotting: return
        # Reset colorbar ref so _draw_hmap_and_ycuts recreates the inset axes fresh.
        # ax_hmap.cla() (called inside) destroys the old inset axes, leaving _cbar
        # pointing at a dead Axes — update_normal() on it crashes on second run.
        self._cbar = None; self.ax_cbar = None
        self._draw_hmap_and_ycuts()

        # ── Block 4: 4 heatmaps (slowest) ────────────────────────────
        _progress(4, '|a_T|² / |a_W|² heatmaps')
        if not self._plotting: return
        _pcm(self.ax_aT_ring, aT_ring,   x_iters, theta_idx, 'viridis',
             '#a8ff78', '|a_T|²  ring',      'Iteration', 'θ')
        m_aT = _pcm(self.ax_aT_all,  aT_all,    x_iters, theta_idx, 'viridis',
             '#a8ff78', '|a_T|²  all (sum)', 'Iteration', 'θ')
        _pcm(self.ax_aW_ring, aW_ring_dB, x_iters, FSR_s, 'hot',
             '#ff8c42', '10·log|a_W|²  ring',      'Iteration', 'FSR μ')
        m_aW = _pcm(self.ax_aW_all,  aW_all_dB,  x_iters, FSR_s, 'hot',
             '#ff8c42', '10·log|a_W|²  all (sum)',  'Iteration', 'FSR μ')
        _add_cbar(self.ax_aT_all, m_aT)
        _add_cbar(self.ax_aW_all, m_aW)
        self.canvas_heat4.draw_idle()

        # show last snapshot in the visualization panels
        _progress(5, 'visualization panels')
        if not self._plotting: return
        last_step = self._hist.shape[2] - 1
        self._vis_frame(last_step)

        # Done — restore all widgets
        self._plotting = False
        _progress(6, 'done')
        self.prog.setValue(100)
        for w in self._plot_widgets:
            w.setEnabled(True)
        self.btn_stop.setEnabled(False)   # stop only active during a run
        self.status.showMessage(f'✅ Plotting complete — {self._hist.shape[2]} snapshots')

    # ------------------------------------------------------------------
    # Focused redraw helpers — no recalculation, use self._slow_cache
    # ------------------------------------------------------------------
    def _draw_comb(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        freq_J       = c['freq_J']
        power_slow_dB= c['power_slow_dB']     # rows in centered μ order (fftshifted)
        mode_indices = c['mode_indices']       # [-N,...,+N] centered μ labels
        one_side     = c['one_side']
        shift_factor = self.spn_fsr_shift.value()
        mode_index   = self.spn_mode_idx.value()   # physical μ, signed
        # Row in the centered-display arrays: μ = -one_side at row 0, μ = 0 at row one_side.
        index        = mode_index + one_side

        ax = self.ax_comb_dB
        ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        for i, m in enumerate(mode_indices):
            # selected tooth bright cyan, others warm amber — both visible on dark bg
            col = '#00e5ff' if i == index else '#c87020'
            lw  = 1.4       if i == index else 0.7
            ax.plot(freq_J + shift_factor*(m+1), power_slow_dB[i], color=col, lw=lw)

        xlim = [-shift_factor*(one_side+1), shift_factor*(one_side+1)]
        ax.set_xlim(xlim)
        ax.set_ylim([power_slow_dB.max()-80, power_slow_dB.max()+5])
        ax.set_title(f'Comb slow-time spectrum (dB)  — click tooth to zoom  |  μ={mode_index}',
                     color='#c8d0e7', fontsize=8, pad=2)
        ax.set_xlabel(f'Freq  (FSR={shift_factor} J)', color=TEXT_DIM, fontsize=7)
        ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.canvas_comb.draw_idle()

    def _draw_zoom(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        freq_J        = c['freq_J']
        power_slow_dB = c['power_slow_dB']      # centered μ rows
        mode_index    = self.spn_mode_idx.value()    # physical μ, signed
        nest_range    = self.spn_nest_range.value()
        one_side      = c['one_side']
        index         = mode_index + one_side

        ax = self.ax_comb_zoom
        ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        ax.plot(freq_J, power_slow_dB[index], color='#4a9eff', lw=1.4)
        ax.set_xlim([-nest_range, nest_range])
        ax.set_title(f'Zoomed spectrum — mode μ={mode_index} (dB)', color='#4a9eff', fontsize=8, pad=2)
        ax.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.canvas_comb.draw_idle()

    def _draw_hmap_and_ycuts(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        nest_range   = self.spn_nest_range.value()
        ycut_freq    = self.spn_ycut_freq.value()
        freq_J       = c['freq_J']
        A_freq       = c['A_freq']
        power_slow_dB= c['power_slow_dB']
        mode_indices = c['mode_indices']

        # recompute trimmed window with current nest_range
        freq_mask    = (freq_J >= -nest_range) & (freq_J <= nest_range)
        freq_trimmed = freq_J[freq_mask]
        pwr_dB_trim  = power_slow_dB[:, freq_mask]

        # y-cut at chosen frequency
        if len(freq_trimmed) > 0:
            freq_idx   = np.argmin(np.abs(freq_trimmed - ycut_freq))
            actual_yf  = freq_trimmed[freq_idx]
            # use A_freq (already fftshifted) for phase — find column in full freq_J
            full_idx   = np.argmin(np.abs(freq_J - ycut_freq))
            pwr_ycut_dB = pwr_dB_trim[:, freq_idx]
            A_ycut      = A_freq[:, full_idx]
            phase_ycut  = np.unwrap(np.angle(A_ycut))
        else:
            actual_yf = ycut_freq
            pwr_ycut_dB = np.zeros(len(mode_indices))
            phase_ycut  = np.zeros(len(mode_indices))

        # heatmap
        self.ax_hmap.cla(); self.ax_hmap.set_facecolor(PANEL_BG)
        if len(freq_trimmed) > 1:
            ext = [freq_trimmed[0], freq_trimmed[-1], mode_indices[0], mode_indices[-1]]
            im  = self.ax_hmap.imshow(pwr_dB_trim[:, ::-1], aspect='auto',
                                      extent=ext, origin='lower', cmap='inferno')
            self.ax_hmap.axvline(actual_yf, color='white', lw=1.0, ls='--', alpha=0.8)
            # horizontal dotted line at the selected mode index
            self.ax_hmap.axhline(self.spn_mode_idx.value(),
                                 color='#87ceeb', lw=2.0, ls=':', alpha=0.9)
            # colorbar: placed as a slim inset INSIDE the right edge of the heatmap
            # bbox_to_anchor=(0.98, 0.02, 0.02, 0.96) keeps it fully within axes bounds
            if self._cbar is None:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
                self.ax_cbar = _inset_axes(self.ax_hmap,
                                           width='3%', height='90%',
                                           loc='center right',
                                           bbox_to_anchor=(0, 0, 1.06, 1),
                                           bbox_transform=self.ax_hmap.transAxes,
                                           borderpad=0)
                self._cbar = self.fig_spec2d.colorbar(im, cax=self.ax_cbar)
                clim = im.get_clim()
                self._cbar.set_ticks([clim[0], clim[1]])
                self._cbar.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
                self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=6)
                for lbl in self._cbar.ax.get_yticklabels():
                    lbl.set_color(TEXT_COL); lbl.set_fontsize(6); lbl.set_clip_on(False)
            else:
                self._cbar.update_normal(im)
                clim = im.get_clim()
                self._cbar.set_ticks([clim[0], clim[1]])
                self._cbar.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
                self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=6, pad=2)
                for lbl in self._cbar.ax.get_yticklabels():
                    lbl.set_color(TEXT_COL); lbl.set_fontsize(6); lbl.set_clip_on(False)
        self.ax_hmap.set_title('2D slow-time spectrum  (all modes × freq)',
                               color='#ff8c42', fontsize=8, pad=2)
        self.ax_hmap.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.set_ylabel('Mode index μ', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_hmap.spines.values(): sp.set_edgecolor('#3a4560')

        # y-cut power
        self.ax_ycut_pwr.cla(); self.ax_ycut_pwr.set_facecolor(PANEL_BG)
        self.ax_ycut_pwr.scatter(mode_indices, pwr_ycut_dB, color='#c8d0e7', s=4)
        self.ax_ycut_pwr.set_title(f'Y-cut power  (freq≈{actual_yf:.3f} J)',
                                   color='#c8d0e7', fontsize=8, pad=2)
        self.ax_ycut_pwr.set_xlabel('Mode μ', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_pwr.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_pwr.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_ycut_pwr.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_ycut_pwr.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        # y-cut phase
        self.ax_ycut_phs.cla(); self.ax_ycut_phs.set_facecolor(PANEL_BG)
        self.ax_ycut_phs.scatter(mode_indices, phase_ycut, color='#c8d0e7', s=4)
        self.ax_ycut_phs.set_title(f'Y-cut phase  (freq≈{actual_yf:.3f} J)',
                                   color='#c8d0e7', fontsize=8, pad=2)
        self.ax_ycut_phs.set_xlabel('Mode μ', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_phs.set_ylabel('Phase (rad)', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_phs.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_ycut_phs.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_ycut_phs.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.canvas_spec2d.draw_idle()

    # ------------------------------------------------------------------
    # Live replot slots — triggered by spinbox changes, no recalc
    # ------------------------------------------------------------------
    def _replot_all(self):
        """Replot all analysis panels without recalculating — triggered by Replot button."""
        if not hasattr(self, '_slow_cache'): return
        self._draw_comb()
        self._draw_zoom()
        self._draw_hmap_and_ycuts()
        self._update_flow()

    def _replot_zoom(self):
        self._draw_comb(); self._draw_zoom()

    def _replot_zoom_and_hmap(self):
        self._draw_zoom(); self._draw_hmap_and_ycuts()

    def _replot_ycut(self):
        self._draw_hmap_and_ycuts()

    def _replot_comb(self):
        self._draw_comb()

    def _on_comb_click(self, event):
        """Click on the comb plot to select the nearest tooth, then replot."""
        if event.inaxes != self.ax_comb_dB: return
        if not hasattr(self, '_slow_cache'): return
        c            = self._slow_cache
        shift_factor = self.spn_fsr_shift.value()
        mode_indices = c['mode_indices']
        centers      = shift_factor * (mode_indices + 1)
        nearest      = int(mode_indices[np.argmin(np.abs(centers - event.xdata))])
        # blockSignals to avoid double-draw from valueChanged; we draw manually below
        self.spn_mode_idx.blockSignals(True)
        self.spn_mode_idx.setValue(nearest)
        self.spn_mode_idx.blockSignals(False)
        self._draw_comb(); self._draw_zoom()
        self._update_flow()

    def _update_flow(self):
        """Redraw only the photon flow panel with the current mode index and vis step."""
        if self._hist is None: return
        step = int(self.spn_vis_step.value())
        step = max(0, min(step, self._hist.shape[2] - 1))
        self._vis_frame(step)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _compute_flow_global_max(self, theta_pick):
        """Compute the maximum arrow magnitude across ALL frames at the given
        theta slice. Used to globally normalize the photon-flow arrows so their
        lengths are comparable across frames.

        Cached on self: recomputing this sweeps every step of history and every
        lattice site, which is expensive for long histories. Re-computed only
        when history or selected θ-slice changes.
        """
        cache_key = (id(self._hist), int(theta_pick))
        if getattr(self, '_flow_max_cache_key', None) == cache_key:
            return self._flow_max_cache_val

        hist  = self._hist
        p     = self._p
        ls    = p.get('linear_state', {})
        NL    = p['NLattice']
        ISite = p['ISite']; OSite = p['OSite']
        h_str = ls.get('h_str', 'IQH_IQH')
        nx0   = ls.get('nx0', 4); ny0 = ls.get('ny0', 4)
        nx1   = ls.get('nx1', 1); ny1 = ls.get('ny1', 1)

        H_mat_full = np.zeros((NL+1, NL+1), dtype=complex)
        H_mat_full[1:, 1:] = p['H_mat']

        NS = hist.shape[2]
        # Subsample: computing flow on every single frame was O(NS) calls to
        # compute_flow, which cost ~100s for NS=100k. The max of a subsampled
        # trace is within a fraction of a percent of the true max — plenty for
        # setting a visual normalization scale. Cap at ≤1000 sample frames.
        stride = max(1, NS // 1000)
        sample_indices = range(0, NS, stride)
        global_max = 0.0
        for si in sample_indices:
            field_mode = hist[:, theta_pick, si]
            fld = np.zeros(NL+1, dtype=complex); fld[1:] = field_mode
            _, _, U, V, _ = compute_flow(
                fld, H_mat_full, NL, ISite, OSite,
                h_str, nx0, ny0, nx1, ny1,
                normalize=False)
            if len(U):
                mx = float(np.sqrt(U**2 + V**2).max())
                if mx > global_max:
                    global_max = mx

        if global_max == 0.0:
            global_max = 1.0   # guard against divide-by-zero when flow is zero everywhere
        self._flow_max_cache_key = cache_key
        self._flow_max_cache_val = global_max
        return global_max

    def _vis_frame(self, step_idx):
        """Draw 2D power and photon flow — Visualization_Figures style, Linear.py colours."""
        if self._hist is None: return
        hist  = self._hist
        p     = self._p
        ls    = p.get('linear_state', {})
        NL    = p['NLattice']
        ISite = p['ISite']; OSite = p['OSite']
        step_idx = max(0, min(step_idx, hist.shape[2]-1))

        a_T_step = hist[:, :, step_idx]        # (NLattice, NumFSRs) complex, θ-space
        NumFSRs  = a_T_step.shape[1]

        # NOTE: historically this picked a THETA index, not a μ mode, using
        # (mode_index + NF//2). Preserving that behavior here so plots match what
        # the old code produced. A proper μ-mode extraction would require FFT'ing
        # to a_W first; that is a separate change.
        theta_pick = int(np.clip(self.spn_mode_idx.value() + NumFSRs // 2, 0, NumFSRs - 1))
        field_mode = a_T_step[:, theta_pick]

        h_str = ls.get('h_str', 'IQH_IQH')
        nx0   = ls.get('nx0', 4); ny0 = ls.get('ny0', 4)
        nx1   = ls.get('nx1', 1); ny1 = ls.get('ny1', 1)
        coords  = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}
        c2n     = {(int(round(v[0])), int(round(v[1]))): k for k, v in coords.items()}

        R_field  = RING_R
        cmap_hot = plt.cm.hot
        # Global history max — makes brightness comparable across frames.
        # Previously this used per-frame max divided by 1.3, which clipped
        # peaks and hid cross-frame intensity differences.
        pmax_global = np.abs(self._hist).max()**2 or 1.
        norm_2d  = plt.Normalize(0, pmax_global)
        phi_fine = np.linspace(0, 2*np.pi, NumFSRs, endpoint=False)
        span_x   = nx1 * nx0;  span_y = ny1 * ny0

        # ── 2D power — ring arc LineCollection traces ──────────────────
        ax = self.ax_vis_2d; ax.cla(); ax.set_facecolor('black')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'2D Power  |  step {step_idx}', color='#c8d0e7', fontsize=8, pad=2)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2230')

        if nx1 > 1 or ny1 > 1:
            from matplotlib.patches import Rectangle
            for i1 in range(nx1):
                for j1 in range(ny1):
                    fc_p = '#444444' if (i1 + j1) % 2 == 0 else '#888888'
                    ax.add_patch(Rectangle((i1*nx0 - 0.5, j1*ny0 - 0.5), nx0, ny0,
                                           linewidth=0, facecolor=fc_p, alpha=0.3))

        for i in range(NL):
            n = i + 1
            xn, yn  = coords[n]
            power_n = np.abs(a_T_step[i, :])**2
            x_arc = xn + R_field * np.cos(phi_fine)
            y_arc = yn + R_field * np.sin(phi_fine)
            pts   = np.array([x_arc, y_arc]).T.reshape(-1, 1, 2)
            segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap_hot, norm=norm_2d, linewidth=2.0, zorder=2)
            lc.set_array(power_n[:-1]); lc.set_rasterized(True)
            ax.add_collection(lc)

            # outline: IN=#4a9eff  OUT=#ff4a6e  others=hot cmap
            out_theta = np.linspace(0, 2*np.pi, 200)
            out_power = np.interp(out_theta, phi_fine, power_n)
            x_out = xn + R_field * np.cos(out_theta)
            y_out = yn + R_field * np.sin(out_theta)
            op   = np.array([x_out, y_out]).T.reshape(-1, 1, 2)
            os_  = np.concatenate([op[:-1], op[1:]], axis=1)
            if n == ISite:
                ol = LineCollection(os_, colors='#4a9eff', linewidths=2.5, alpha=0.7, zorder=3)
            elif n == OSite:
                ol = LineCollection(os_, colors='#ff4a6e', linewidths=2.5, alpha=0.7, zorder=3)
            else:
                ol = LineCollection(os_, cmap=cmap_hot, norm=norm_2d, linewidths=0.8, zorder=3)
                ol.set_array(out_power[:-1])
            ol.set_rasterized(True)
            ax.add_collection(ol)

        is_zz  = (h_str == 'A_zigzag')
        is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
        if is_zz or is_cyl:
            all_x = [v[0] for v in coords.values()]
            all_y = [v[1] for v in coords.values()]
            pad = R_field * 2
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        else:
            ax.set_xlim(-1, span_x); ax.set_ylim(-1, span_y)
        ax = self.ax_vis_flow; ax.cla(); ax.set_facecolor('black')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Photon Flow  μ={self.spn_mode_idx.value()}  |  step {step_idx}',
                     color='#c8d0e7', fontsize=8, pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor('white'); sp.set_linewidth(1.5)

        H_mat_flow = np.zeros((NL+1, NL+1), dtype=complex)
        H_mat_flow[1:, 1:] = p['H_mat']
        fld = np.zeros(NL+1, dtype=complex); fld[1:] = field_mode
        # Get RAW arrow field (normalize=False), then divide by the history-wide
        # max so arrow length is comparable across frames. The cached global max
        # is computed once per (hist, θ-slice) and reused.
        Xq, Yq, U, V, cols_f = compute_flow(
            fld, H_mat_flow, NL, ISite, OSite,
            h_str, nx0, ny0, nx1, ny1,
            normalize=False)
        flow_gmax = self._compute_flow_global_max(theta_pick)
        if flow_gmax > 0:
            U = U / flow_gmax
            V = V / flow_gmax
        if len(Xq) and np.sqrt(U**2 + V**2).max() > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ax.quiver(Xq, Yq, U, V, color=cols_f,
                          scale=0.6, scale_units='xy', width=0.01, pivot='middle', zorder=3)

        is_zz  = (h_str == 'A_zigzag')
        is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
        if is_zz:
            ax.set_xlim(-0.5, 2*nx0+0.5); ax.set_ylim(-0.5, 2*ny0+0.5)
        elif is_cyl:
            ax.autoscale_view()
        else:
            ax.set_xlim(-1, span_x); ax.set_ylim(-1, span_y)

        # ── 3D power — ring line traces, black bg, white/colored ───────
        ax3 = self.ax_vis_3d; ax3.cla()
        ax3.set_facecolor('black')
        ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('none')
        ax3.yaxis.pane.set_edgecolor('none')
        ax3.zaxis.pane.set_edgecolor('none')
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
        ax3.set_axis_off()
        ax3.set_box_aspect([1, 1, 0.5])
        ax3.view_init(elev=80, azim=270)
        ax3.set_title(f'3D Power  |  step {step_idx}', color='#c8d0e7', fontsize=8, pad=2)

        pmax_global = np.abs(self._hist).max()**2 or 1.
        for i in range(NL):
            n = i + 1
            xn, yn  = coords[n]
            power_n = np.abs(a_T_step[i, :])**2 / pmax_global
            x3 = xn + R_field * np.cos(phi_fine)
            y3 = yn + R_field * np.sin(phi_fine)
            z3 = power_n
            x3 = np.append(x3, x3[0]); y3 = np.append(y3, y3[0]); z3 = np.append(z3, z3[0])
            tc = '#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else 'white'
            line = ax3.plot(x3, y3, z3, color=tc, linewidth=0.8)
            line[0].set_rasterized(True)

        self.canvas_2dpow.draw_idle()
        self.canvas_flow.draw_idle()
        self.canvas_3d.draw_idle()

    def _vis_plot(self):
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.')
            return
        step = self.spn_vis_step.value()
        self.spn_vis_step.setRange(0, self._hist.shape[2]-1)
        step = min(step, self._hist.shape[2]-1)
        self._vis_frame(step)
        self.lbl_vis_status.setText(f'Plotted step {step}.')

    def _vis_save_step(self):
        """Save the three visualization figures for the currently plotted step."""
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.'); return

        # ── Protection: disable visualization buttons while saving.
        _snapshots = {
            'plot': (self.btn_vis_plot, self.btn_vis_plot.isEnabled()),
            'save': (self.btn_vis_save, self.btn_vis_save.isEnabled()),
            'gif':  (self.btn_vis_gif,  self.btn_vis_gif.isEnabled()),
            'mp4':  (self.btn_vis_mp4,  self.btn_vis_mp4.isEnabled()),
        }
        for _btn, _ in _snapshots.values():
            _btn.setEnabled(False)
        self.lbl_vis_status.setText('💾  Saving...')
        self.lbl_vis_status.setStyleSheet('color:#ffcc00;font-size:11px;font-weight:bold;')
        QApplication.processEvents()

        try:
            folder = self._p.get('session_folder')
            if not folder:
                folder = QFileDialog.getExistingDirectory(
                    self, 'Choose save folder', os.path.expanduser('~'))
                if not folder:
                    return
            os.makedirs(folder, exist_ok=True)

            step = self.spn_vis_step.value()
            mu   = self.spn_mode_idx.value()

            for fig, name in [
                (self.fig_2dpow, f'04_2d_power_step{step}'),
                (self.fig_flow,  f'07_photon_flow_step{step}_mu{mu}'),
                (self.fig_3d,    f'08_3d_power_step{step}'),
            ]:
                for fmt, dpi in [('png', 200), ('svg', 150)]:
                    buf = io.BytesIO()
                    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    with open(os.path.join(folder, f'{name}.{fmt}'), 'wb') as fh:
                        fh.write(buf.getvalue())

            self.lbl_vis_status.setText(f'✅  Saved step {step} → {os.path.basename(folder)}')
            self.lbl_vis_status.setStyleSheet('color:#a8ff78;font-size:11px;')

        except Exception as exc:
            self.lbl_vis_status.setText(f'❌  Save failed: {exc}')
            self.lbl_vis_status.setStyleSheet('color:#ff4a6e;font-size:11px;font-weight:bold;')
            import traceback; traceback.print_exc()

        finally:
            for _btn, _was_enabled in _snapshots.values():
                _btn.setEnabled(_was_enabled)

    def _on_exit(self):
        """Interrupt any ongoing plotting and close the window."""
        self._plotting = False
        self._stop.set()   # also stop any running simulation
        self.close()

    def _vis_nav(self, direction):
        """Close this window and open a new SlowTimeWindow seeded from
        step_idx ± (direction * nav_step). `direction` is ±1; actual jump size
        is controlled by spn_vis_nav_step."""
        parent = self.parent()
        if parent is None or self._p is None: self.close(); return
        if not hasattr(parent, '_results') or parent._results is None:
            self.close(); return
        nav_step = int(self.spn_vis_nav_step.value())
        current  = self._p.get('step_idx', 0)
        n_steps  = parent._results['a_T'].shape[2]
        new_idx  = int(np.clip(current + direction * nav_step, 0, n_steps - 1))
        parent.sld_probe.setValue(new_idx)
        parent._open_slow_time()
        self.close()

    def _vis_export_gif(self):
        self._vis_export('gif')

    def _vis_export_mp4(self):
        self._vis_export('mp4')

    def _vis_export(self, fmt):
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.'); return

        NS = self._hist.shape[2]

        # ── Ask user for frame range (no folder dialog) ─────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Export range')
        dlg.setStyleSheet(STYLE)
        dlg.setMinimumWidth(300)
        fl = QGridLayout(dlg); fl.setSpacing(6)

        def _lbl(t): lb = QLabel(t); lb.setStyleSheet('color:#c8d0e7;font-size:13px;'); return lb

        spn_s = QSpinBox(); spn_s.setRange(0, NS-1); spn_s.setValue(max(0, NS-1001))
        spn_e = QSpinBox(); spn_e.setRange(0, NS-1); spn_e.setValue(NS-1)
        spn_t = QSpinBox(); spn_t.setRange(1, max(1, NS//2)); spn_t.setValue(10)
        spn_t.setToolTip('Save every N-th snapshot as a frame')
        # FPS: playback rate. Lower = slower playback (each frame on screen longer).
        # GIF tends to look better at lower fps; MP4 handles higher fps cleanly.
        spn_fps = QSpinBox(); spn_fps.setRange(1, 60)
        spn_fps.setValue(5 if fmt == 'gif' else 10)
        spn_fps.setToolTip('Playback frame rate (frames per second)')

        fl.addWidget(_lbl('Start step:'),    0, 0); fl.addWidget(spn_s,   0, 1)
        fl.addWidget(_lbl('End step:'),      1, 0); fl.addWidget(spn_e,   1, 1)
        fl.addWidget(_lbl('Step interval:'), 2, 0); fl.addWidget(spn_t,   2, 1)
        fl.addWidget(_lbl('Playback FPS:'),  3, 0); fl.addWidget(spn_fps, 3, 1)

        lbl_n = QLabel('')
        lbl_n.setStyleSheet('color:#4a5270;font-size:12px;')
        fl.addWidget(lbl_n, 4, 0, 1, 2)

        def _update_n(*_):
            s = spn_s.value(); e = spn_e.value(); t = spn_t.value(); f = spn_fps.value()
            n = max(0, (e - s) // t + 1) if e >= s else 0
            dur = n / f if f > 0 else 0
            lbl_n.setText(f'→  {n} frames  |  duration ≈ {dur:.1f} s at {f} fps')
        spn_s.valueChanged.connect(_update_n)
        spn_e.valueChanged.connect(_update_n)
        spn_t.valueChanged.connect(_update_n)
        spn_fps.valueChanged.connect(_update_n)
        _update_n()

        btn_row = QHBoxLayout()
        btn_ok  = QPushButton('Export'); btn_ok.clicked.connect(dlg.accept)
        btn_can = QPushButton('Cancel'); btn_can.clicked.connect(dlg.reject)
        btn_row.addStretch(); btn_row.addWidget(btn_ok); btn_row.addWidget(btn_can)
        fl.addLayout(btn_row, 5, 0, 1, 2)

        if dlg.exec_() != QDialog.Accepted: return

        s_start = spn_s.value(); s_end = spn_e.value(); s_step = spn_t.value()
        fps_val = spn_fps.value()
        if s_end < s_start: s_start, s_end = s_end, s_start
        steps = list(range(s_start, s_end + 1, s_step))
        if not steps:
            self.lbl_vis_status.setText('No frames in range.'); return

        # Save into session folder (same as the other files)
        folder = self._p.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)
        mu    = self.spn_mode_idx.value()
        fname = f'movie_s{s_start}_e{s_end}_step{s_step}_fps{fps_val}_mu{mu}.{fmt}'
        out   = os.path.join(folder, fname)

        import matplotlib.animation as animation
        self.lbl_vis_status.setText(f'Exporting {len(steps)} frames…')
        self.canvas_2dpow.draw_idle(); QApplication.processEvents()

        p       = self._p; ls = p.get('linear_state', {})
        NL      = p['NLattice']; ISite = p['ISite']; OSite = p['OSite']
        NumFSRs = self._hist.shape[1]
        # See note in _vis_frame: this is a THETA index, preserved from the old behavior.
        theta_pick = int(np.clip(self.spn_mode_idx.value() + NumFSRs // 2, 0, NumFSRs - 1))
        h_str = ls.get('h_str', 'IQH_IQH')
        nx0   = ls.get('nx0', 4); ny0 = ls.get('ny0', 4)
        nx1   = ls.get('nx1', 1); ny1 = ls.get('ny1', 1)
        coords  = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}
        c2n     = {(int(round(v[0])), int(round(v[1]))): k for k, v in coords.items()}
        H_mat   = np.zeros((NL+1, NL+1), dtype=complex); H_mat[1:, 1:] = p['H_mat']
        phi_e   = np.linspace(0, 2*np.pi, NumFSRs, endpoint=False)
        span_x  = nx1 * nx0;  span_y = ny1 * ny0
        R_e     = RING_R
        cmap_h  = plt.cm.hot

        fig_exp = plt.figure(figsize=(8, 5), facecolor=DARK_BG)
        ax1 = fig_exp.add_subplot(2, 2, 1)
        ax2 = fig_exp.add_subplot(2, 2, 3)
        ax3 = fig_exp.add_subplot(1, 2, 2, projection='3d')
        fig_exp.subplots_adjust(hspace=0.25, wspace=0.15,
                                left=0.02, right=0.98, top=0.94, bottom=0.03)
        pmax_global_e = np.abs(self._hist).max()**2 or 1.

        # Global flow max across the whole history at the selected θ-slice.
        # Computed once (cached on self) so arrow lengths are comparable
        # across all frames of the exported movie.
        flow_gmax_e = self._compute_flow_global_max(theta_pick)

        def _frame(si):
            ax1.cla(); ax2.cla(); ax3.cla()
            a_T = self._hist[:, :, si]
            # Global 2D power normalization — consistent across frames.
            norm_e = plt.Normalize(0, pmax_global_e)
            field_m = a_T[:, theta_pick]

            # ── 2D power ─────────────────────────────────────────────
            ax1.set_facecolor('black'); ax1.set_aspect('equal')
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title(f'2D Power  step={si}', color='#c8d0e7', fontsize=8)
            if nx1 > 1 or ny1 > 1:
                from matplotlib.patches import Rectangle as _R
                for i1 in range(nx1):
                    for j1 in range(ny1):
                        fc_p = '#444444' if (i1+j1)%2==0 else '#888888'
                        ax1.add_patch(_R((i1*nx0-0.5, j1*ny0-0.5), nx0, ny0,
                                         linewidth=0, facecolor=fc_p, alpha=0.3))
            for i in range(NL):
                n = i+1; xn, yn = coords[n]
                pn = np.abs(a_T[i, :])**2
                pts = np.array([xn+R_e*np.cos(phi_e), yn+R_e*np.sin(phi_e)]).T.reshape(-1,1,2)
                seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(seg, cmap=cmap_h, norm=norm_e, linewidth=2.0, zorder=2)
                lc.set_array(pn[:-1]); ax1.add_collection(lc)
                ot = np.linspace(0, 2*np.pi, 200)
                op = np.interp(ot, phi_e, pn)
                osp = np.array([xn+R_e*np.cos(ot), yn+R_e*np.sin(ot)]).T.reshape(-1,1,2)
                oss = np.concatenate([osp[:-1], osp[1:]], axis=1)
                if n == ISite:   ol = LineCollection(oss, colors='#4a9eff', linewidths=2.5, alpha=0.7, zorder=3)
                elif n == OSite: ol = LineCollection(oss, colors='#ff4a6e', linewidths=2.5, alpha=0.7, zorder=3)
                else:
                    ol = LineCollection(oss, cmap=cmap_h, norm=norm_e, linewidths=0.8, zorder=3)
                    ol.set_array(op[:-1])
                ax1.add_collection(ol)
            _is_zz  = (h_str == 'A_zigzag')
            _is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
            if _is_zz or _is_cyl:
                all_x = [v[0] for v in coords.values()]
                all_y = [v[1] for v in coords.values()]
                pad = R_e * 2
                ax1.set_xlim(min(all_x) - pad, max(all_x) + pad)
                ax1.set_ylim(min(all_y) - pad, max(all_y) + pad)
            else:
                ax1.set_xlim(-1, span_x); ax1.set_ylim(-1, span_y)

            # ── Photon flow ──────────────────────────────────────────
            ax2.set_facecolor('black'); ax2.set_aspect('equal')
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title(f'Photon Flow μ={self.spn_mode_idx.value()}  step={si}', color='#c8d0e7', fontsize=8)
            for sp in ax2.spines.values(): sp.set_edgecolor('white'); sp.set_linewidth(1.5)
            fld_e = np.zeros(NL+1, dtype=complex); fld_e[1:] = field_m
            # Global-normalize flow: raw U,V / history-wide max
            Xq_, Yq_, U_, V_, cf_ = compute_flow(
                fld_e, H_mat, NL, ISite, OSite, h_str, nx0, ny0, nx1, ny1,
                normalize=False)
            if flow_gmax_e > 0:
                U_ = U_ / flow_gmax_e
                V_ = V_ / flow_gmax_e
            ax2.quiver(Xq_, Yq_, U_, V_, color=cf_,
                       scale=0.6, scale_units='xy', width=0.01, pivot='middle', zorder=3)
            _is_zz  = (h_str == 'A_zigzag')
            _is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
            if _is_zz:   ax2.set_xlim(-0.5, 2*nx0+0.5); ax2.set_ylim(-0.5, 2*ny0+0.5)
            elif _is_cyl: ax2.autoscale_view()
            else:         ax2.set_xlim(-1, span_x); ax2.set_ylim(-1, span_y)

            # ── 3D ring traces — black bg, white/colored ─────────────
            ax3.set_facecolor('black')
            ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
            ax3.xaxis.pane.set_edgecolor('none')
            ax3.yaxis.pane.set_edgecolor('none')
            ax3.zaxis.pane.set_edgecolor('none')
            ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
            ax3.set_axis_off(); ax3.set_box_aspect([1, 1, 0.5])
            ax3.view_init(elev=80, azim=270)
            ax3.set_title(f'3D Power  step={si}', color='#c8d0e7', fontsize=8)
            for i in range(NL):
                n = i+1; xn, yn = coords[n]
                pn = np.abs(a_T[i, :])**2 / pmax_global_e
                x3 = xn + R_e*np.cos(phi_e); y3 = yn + R_e*np.sin(phi_e); z3 = pn
                x3 = np.append(x3, x3[0]); y3 = np.append(y3, y3[0]); z3 = np.append(z3, z3[0])
                tc = '#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else 'white'
                ax3.plot(x3, y3, z3, color=tc, linewidth=0.8)
            return []

        if fmt == 'gif':
            ani = animation.FuncAnimation(fig_exp, _frame, frames=steps, blit=False)
            ani.save(out, writer='pillow', fps=fps_val)
            plt.close(fig_exp)
        else:
            # Use imageio + imageio-ffmpeg (bundles its own ffmpeg, no system install needed)
            try:
                import imageio
                from PIL import Image as _PILImage
                frames_arr = []
                for si in steps:
                    _frame(si)
                    buf = io.BytesIO()
                    fig_exp.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    frames_arr.append(np.array(_PILImage.open(buf).convert('RGB')))
                plt.close(fig_exp)
                imageio.mimwrite(out, frames_arr, fps=fps_val, quality=8)
            except ImportError:
                plt.close(fig_exp)
                self.lbl_vis_status.setText(
                    '❌ MP4 needs imageio: pip install imageio imageio-ffmpeg')
                return
        self.lbl_vis_status.setText(f'Saved → {os.path.basename(out)}')

    # ------------------------------------------------------------------
    def _on_save(self):
        if self._hist is None:
            self.status.showMessage('No data yet — run first.')
            return

        # ── Picker dialog: choose which figures + formats + data file ──
        # Each figure can be 5–50 MB as SVG (heatmaps rasterize worst),
        # so letting the user deselect saves a lot of disk + time.
        mu    = self.spn_mode_idx.value()
        fsr   = self.spn_fsr_shift.value()
        nest  = self.spn_nest_range.value()
        ycut  = self.spn_ycut_freq.value()
        def _flt(v): return f'{v:.3g}'.replace('.', 'p').replace('-', 'm')
        mu_str   = f'_mu{mu}'
        fsr_str  = f'_FSR{_flt(fsr)}'
        nest_str = f'_nest{_flt(nest)}'
        ycut_str = f'_ycut{_flt(ycut)}'

        # Figure list — same as the old save loop
        all_figs = [
            (self.fig_ind,    '01_sweep_indicator',                                 'Sweep indicator (4 power panels)'),
            (self.fig_heat4,  '02_heatmaps',                                        'Heatmaps (|a_T|² and |a_W|² ring+all) — large'),
            (self.fig_comb,   f'03_comb_spectrum{mu_str}{fsr_str}',                 'Comb spectrum + zoomed tooth'),
            (self.fig_2dpow,  '04_2d_power',                                        '2D power (rings)'),
            (self.fig_esa,    '05_esa_oscilloscope',                                'ESA + oscilloscope'),
            (self.fig_spec2d, f'06_2d_spectrum{mu_str}{fsr_str}{nest_str}{ycut_str}','2D slow-time spectrum + y-cuts — large'),
            (self.fig_flow,   f'07_photon_flow{mu_str}',                            'Photon flow'),
            (self.fig_3d,     '08_3d_power',                                        '3D power'),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle('Save slow-time analysis')
        dlg.setStyleSheet(STYLE)
        dlg.setMinimumWidth(440)
        dl = QVBoxLayout(dlg); dl.setSpacing(8); dl.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel('Select what to save:')
        hdr.setStyleSheet('color:#c8d0e7;font-size:13px;font-weight:bold;')
        dl.addWidget(hdr)

        # ── Figures ──────────────────────────────────────────────────
        gb_figs = QGroupBox('Figures')
        gb_figs.setStyleSheet('QGroupBox{color:#c8d0e7;}')
        vf = QVBoxLayout(gb_figs); vf.setSpacing(3)

        # "Select all" toggle
        from PyQt5.QtWidgets import QCheckBox
        chk_all = QCheckBox('(toggle all figures)')
        chk_all.setChecked(True)
        chk_all.setStyleSheet('color:#8892b0;font-size:11px;font-style:italic;')
        vf.addWidget(chk_all)

        fig_checks = []
        for fig, name, desc in all_figs:
            chk = QCheckBox(f'{name}  —  {desc}')
            chk.setChecked(True)
            chk.setStyleSheet('color:#c8d0e7;font-size:12px;')
            vf.addWidget(chk)
            fig_checks.append((chk, fig, name, desc))

        def _toggle_all(state):
            for chk, _, _, _ in fig_checks:
                chk.setChecked(state == Qt.Checked)
        chk_all.stateChanged.connect(_toggle_all)
        dl.addWidget(gb_figs)

        # ── Formats ──────────────────────────────────────────────────
        gb_fmt = QGroupBox('Formats')
        hf = QHBoxLayout(gb_fmt); hf.setSpacing(12)
        chk_png = QCheckBox('PNG  (raster, small)'); chk_png.setChecked(True)
        chk_svg = QCheckBox('SVG  (vector, large — heatmaps especially)'); chk_svg.setChecked(True)
        for c in (chk_png, chk_svg): c.setStyleSheet('color:#c8d0e7;font-size:12px;')
        hf.addWidget(chk_png); hf.addWidget(chk_svg); hf.addStretch()
        dl.addWidget(gb_fmt)

        # ── Data file ────────────────────────────────────────────────
        gb_dat = QGroupBox('Data')
        hd = QHBoxLayout(gb_dat); hd.setSpacing(12)
        chk_npz = QCheckBox(f'slowtime_data.npz  (full hist array — can be 100+ MB)')
        chk_npz.setChecked(True); chk_npz.setStyleSheet('color:#c8d0e7;font-size:12px;')
        hd.addWidget(chk_npz); hd.addStretch()
        dl.addWidget(gb_dat)

        # ── Buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_ok  = QPushButton('💾  Save selected'); btn_ok.clicked.connect(dlg.accept)
        btn_can = QPushButton('Cancel');             btn_can.clicked.connect(dlg.reject)
        btn_row.addStretch(); btn_row.addWidget(btn_ok); btn_row.addWidget(btn_can)
        dl.addLayout(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            self.status.showMessage('Save cancelled.')
            return

        # Collect selections
        selected_figs = [(fig, name) for chk, fig, name, _ in fig_checks if chk.isChecked()]
        want_png = chk_png.isChecked()
        want_svg = chk_svg.isChecked()
        want_npz = chk_npz.isChecked()
        if not selected_figs and not want_npz:
            self.status.showMessage('Nothing selected — save cancelled.')
            return
        if not want_png and not want_svg and selected_figs:
            self.status.showMessage('No format selected — save cancelled.')
            return

        # ── Protection: disable all action buttons + show status while saving.
        # SlowTimeWindow saves up to 8 figures (PNG+SVG each) plus a potentially
        # large hist .npz — this can take many seconds. processEvents() forces
        # the UI to repaint the disabled/amber state before blocking I/O starts.
        _snapshots = {
            'run':      (self.btn_run,      self.btn_run.isEnabled()),
            'stop':     (self.btn_stop,     self.btn_stop.isEnabled()),
            'save':     (self.btn_save,     self.btn_save.isEnabled()),
            'lyap':     (self.btn_lyapunov, self.btn_lyapunov.isEnabled()),
            'modal':    (self.btn_modal,    self.btn_modal.isEnabled()),
            'replot':   (self.btn_replot,   self.btn_replot.isEnabled()),
            'vis_plot': (self.btn_vis_plot, self.btn_vis_plot.isEnabled()),
            'vis_save': (self.btn_vis_save, self.btn_vis_save.isEnabled()),
            'vis_gif':  (self.btn_vis_gif,  self.btn_vis_gif.isEnabled()),
            'vis_mp4':  (self.btn_vis_mp4,  self.btn_vis_mp4.isEnabled()),
        }
        for _btn, _ in _snapshots.values():
            _btn.setEnabled(False)
        self.status.showMessage('💾  Saving slow-time analysis... (please wait)')
        QApplication.processEvents()

        try:
            folder = self._p.get('session_folder')
            if not folder:
                folder = QFileDialog.getExistingDirectory(
                    self, 'Choose save folder', os.path.expanduser('~'))
                if not folder:
                    return   # finally block will restore buttons
            os.makedirs(folder, exist_ok=True)

            total = len(selected_figs)
            for i_fig, (fig, name) in enumerate(selected_figs, start=1):
                # Progress update so the user knows we're not hung
                self.status.showMessage(f'💾  Saving {i_fig}/{total}: {name}…')
                QApplication.processEvents()

                if want_png:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    with open(os.path.join(folder, f'{name}.png'), 'wb') as fh:
                        fh.write(buf.getvalue())

                if want_svg:
                    # SVG — rasterize artists only for the heatmap figure
                    snapshot = []
                    if fig is self.fig_heat4:
                        for ax in fig.get_axes():
                            for art in ax.get_children():
                                if hasattr(art, 'set_rasterized'):
                                    snapshot.append((art, art.get_rasterized()))
                                    art.set_rasterized(True)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='svg', dpi=150, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    for art, was in snapshot:
                        art.set_rasterized(was)
                    with open(os.path.join(folder, f'{name}.svg'), 'wb') as fh:
                        fh.write(buf.getvalue())

            # ── Data ─────────────────────────────────────────────────────────
            if want_npz and self._hist is not None:
                self.status.showMessage('💾  Saving slowtime_data.npz (may be large)…')
                QApplication.processEvents()
                p = self._p
                # Round bookkeeping: hist contains only the LAST round's snapshots
                # (incremental-run design). We save the per-round NIter list so
                # downstream analysis can reconstruct the absolute time axis even
                # when NIter varied between rounds.
                round_niters_arr = np.asarray(self._round_niters, dtype=np.int64)
                n_rounds_total   = int(self._round_count)
                total_evolved    = int(round_niters_arr.sum()) if len(round_niters_arr) else 0
                # steps BEFORE the first frame of hist (i.e. sum over discarded rounds).
                # hist comes from the FINAL round, so subtract the last round's NIter.
                if len(round_niters_arr):
                    steps_before_hist = int(round_niters_arr[:-1].sum())
                else:
                    steps_before_hist = 0
                np.savez(os.path.join(folder, 'slowtime_data.npz'),
                         hist              = self._hist,
                         FSRVec            = p['FSRVec'],
                         every             = np.array([1]),     # always 1 now
                         step_idx          = np.array([p['step_idx']]),
                         detuning          = np.array([p['detuning']]),
                         F                 = np.array([p['F']]),
                         kin               = np.array([p['kin']]),
                         kex               = np.array([p['kex']]),
                         D2                = np.array([p['D2']]),
                         TStep             = np.array([p['TStep']]),
                         n_rounds_total    = np.array([n_rounds_total]),
                         round_niters      = round_niters_arr,
                         total_evolved     = np.array([total_evolved]),
                         steps_before_hist = np.array([steps_before_hist]))

            self.status.showMessage(f'✅ Saved → {folder}')

        except Exception as exc:
            self.status.showMessage(f'❌  Save failed: {exc}')
            import traceback; traceback.print_exc()

        finally:
            # Restore previous button states
            for _btn, _was_enabled in _snapshots.values():
                _btn.setEnabled(_was_enabled)


# Attach continuation methods from mixin onto SlowTimeWindow
for _name, _method in _SlowTimeWindowMethods.__dict__.items():
    if callable(_method) and not _name.startswith('__'):
        setattr(SlowTimeWindow, _name, _method)
del _SlowTimeWindowMethods


# =====================================================================
class NonlinearWindow(QMainWindow):
    def __init__(self, linear_state: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Nonlinear Comb Simulator  --  fed from Linear Explorer')
        self.setMinimumSize(1600, 840)
        self.setStyleSheet(STYLE)

        ls = linear_state;  self._ls = ls
        self._NLattice = ls['NL']
        self._is_cyl   = ls.get('is_cyl', False)
        self._h_str    = ls['h_str']

        self._param_rows = []
        self._h_sweep    = set()
        self._q          = queue.Queue()
        self._stop_flag  = threading.Event()
        self._poll_timer = QTimer(self);  self._poll_timer.setInterval(80)
        self._poll_timer.timeout.connect(self._poll_queue)

        self._xv = self._pump_n = self._side_n = self._spec_dB = None
        self._pump_n_all = self._side_n_all = None
        self._FSRVec = self._PumpFSR = None;  self._n_steps = 0
        self._hmap_img = self._line_pump = self._line_side = None
        self._vline_pump = self._vline_side = None
        self._sched_vline = self._sched_vline_psi = self._sched_vline_h = None
        self._spec_drag = False
        self._results   = None

        self._build_ui()

        # Draw lattice immediately so rings are visible on open
        self._draw_lattice()
        # These are NOT removable since they define the baseline on-site potential.
        # User can switch them to Sweep or Design to ramp them.
        for site, delta in sorted(ls.get('heaters', {}).items()):
            self._add_param_row(
                f'Heater {site}  (inherited)', f'heater_{site}',
                default_fixed=delta,
                default_from=delta, default_to=delta + 3.0,
                removable=False)

    # ------------------------------------------------------------------
    def showEvent(self, event):
        """Redraw schedule preview when window becomes visible (ensures canvas is ready)."""
        super().showEvent(event)
        self._draw_sched_preview()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget();  self.setCentralWidget(central)
        root = QVBoxLayout(central);  root.setSpacing(4)
        root.setContentsMargins(6,6,6,6)

        t = QLabel('NONLINEAR COMB SIMULATOR  --  fed from Linear Lattice Explorer')
        t.setFont(QFont('Courier New', 13, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(t)
        sep = QFrame();  sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;');  root.addWidget(sep)

        self._splitter = QSplitter(Qt.Horizontal)
        splitter = self._splitter

        # ── Left: schedule preview (full height) ──────────────────────
        lw = QWidget();  ll = QVBoxLayout(lw);  ll.setContentsMargins(0,0,0,0)
        self.fig_sched    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_sched = FigureCanvas(self.fig_sched)
        self.canvas_sched.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ll.addWidget(self.canvas_sched)

        # ── Middle: pump power (top) + comb power (bottom) ────────────
        mw = QWidget();  ml = QVBoxLayout(mw);  ml.setContentsMargins(0,0,0,0)
        self.fig_plots    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_plots = FigureCanvas(self.fig_plots)
        self.canvas_plots.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs = self.fig_plots.add_gridspec(2, 2, hspace=0.52, wspace=0.38,
             left=0.09, right=0.97, top=0.93, bottom=0.10)
        self.ax_pump     = self.fig_plots.add_subplot(gs[0, 0])
        self.ax_side     = self.fig_plots.add_subplot(gs[1, 0])
        self.ax_pump_all = self.fig_plots.add_subplot(gs[0, 1])
        self.ax_side_all = self.fig_plots.add_subplot(gs[1, 1])
        for ax,col,ttl in [
            (self.ax_pump,    '#4a9eff','Pump power  (output ring)'),
            (self.ax_side,    '#ff4a6e','Comb power excl. pump  (output ring)'),
            (self.ax_pump_all,'#4a9eff','Pump power  (all rings)'),
            (self.ax_side_all,'#ff4a6e','Comb power excl. pump  (all rings)')]:
            ax.set_facecolor(PANEL_BG);  ax.set_title(ttl,color=col,fontsize=8,pad=3)
            ax.tick_params(colors=TEXT_COL,labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560');  sp.set_linewidth(1.2)
            ax.grid(True,color=GRID_COL,linewidth=0.4)
        self.canvas_plots.mpl_connect('button_press_event',   self._on_spec_press)
        self.canvas_plots.mpl_connect('motion_notify_event',  self._on_spec_move)
        self.canvas_plots.mpl_connect('button_release_event',
                                       lambda e: setattr(self,'_spec_drag',False))
        ml.addWidget(self.canvas_plots)

        # keep canvas_lat alive for _draw_lattice (off-screen, not in layout)
        self.fig_lat    = Figure(facecolor=DARK_BG)
        self.canvas_lat = FigureCanvas(self.fig_lat)
        self.ax_lat     = self.fig_lat.add_subplot(111)
        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]);  self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values(): sp.set_edgecolor('#3a4560')
        self.fig_lat.subplots_adjust(left=0.01,right=0.99,top=0.97,bottom=0.01)

        # ── Right: a_T and a_W heatmaps (2×2) ────────────────────────
        rw = QWidget();  rl = QVBoxLayout(rw);  rl.setContentsMargins(0,0,0,0)
        self.fig_heat    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.canvas_heat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs_h = self.fig_heat.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
               left=0.10, right=0.97, top=0.93, bottom=0.08)
        self.ax_aT_out = self.fig_heat.add_subplot(gs_h[0, 0])
        self.ax_aT_all = self.fig_heat.add_subplot(gs_h[0, 1])
        self.ax_aW_out = self.fig_heat.add_subplot(gs_h[1, 0])
        self.ax_aW_all = self.fig_heat.add_subplot(gs_h[1, 1])
        for ax, col, ttl in [
            (self.ax_aT_out, '#a8ff78', '|a_T|²  (output ring)'),
            (self.ax_aT_all, '#a8ff78', '|a_T|²  (all rings)'),
            (self.ax_aW_out, '#ff8c42', '10·log|a_W|²  (output ring)'),
            (self.ax_aW_all, '#ff8c42', '10·log|a_W|²  (all rings)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=7)
        self.ax_aT_out.set_ylabel('θ index', color=TEXT_DIM, fontsize=7)
        self.ax_aT_all.set_ylabel('θ index', color=TEXT_DIM, fontsize=7)
        self.ax_aW_out.set_ylabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self.ax_aW_all.set_ylabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self._heat_imgs = {}   # stores imshow objects for fast update
        self._heat_vlines = {}  # vertical line overlays on heatmaps

        # cross-section plots (below heatmaps, update with probe)
        self.fig_xsec    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_xsec = FigureCanvas(self.fig_xsec)
        self.canvas_xsec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.canvas_xsec.setMinimumHeight(200)
        gs_x = self.fig_xsec.add_gridspec(1, 2, wspace=0.4,
               left=0.10, right=0.97, top=0.88, bottom=0.18)
        self.ax_xsec_aW = self.fig_xsec.add_subplot(gs_x[0])
        self.ax_xsec_aT = self.fig_xsec.add_subplot(gs_x[1])
        for ax, col, ttl in [
            (self.ax_xsec_aW, '#ff8c42', '10·log|a_W|²  at step  (osite)'),
            (self.ax_xsec_aT, '#a8ff78', '|a_T|²  at step  (osite)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4)
        self.ax_xsec_aW.set_xlabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self.ax_xsec_aT.set_xlabel('θ index', color=TEXT_DIM, fontsize=7)

        rl.addWidget(self.canvas_heat)
        rl.addWidget(self.canvas_xsec)
        rw.setLayout(rl)

        # ── Far right: slow time analysis panel (hidden until triggered) ──
        splitter.addWidget(lw);  splitter.addWidget(mw)
        splitter.addWidget(rw)
        self._splitter = splitter
        splitter.setSizes([480, 560, 580])
        root.addWidget(splitter,stretch=1)
        root.addWidget(self._build_controls())
        self.status = QStatusBar();  self.setStatusBar(self.status)
        self.status.showMessage(
            f'Ready  |  H_mat inherited  |  ' +
            (f'JAX {jax.__version__} on {jax.default_backend()}' if JAX_AVAILABLE else 'JAX not installed — NumPy fallback active'))

    # ------------------------------------------------------------------
    def _build_controls(self):
        w = QWidget();  row = QHBoxLayout(w)
        row.setSpacing(6);  row.setContentsMargins(0,0,0,0)

        # ── 1. Inherited ──────────────────────────────────────────────
        gi  = QGroupBox('Inherited from Linear Explorer')
        il  = QGridLayout(gi);  il.setSpacing(5)
        ls  = self._ls;  h = ls['h_str']

        def _ro(text, col='#c8d0e7'):
            l = QLabel(text);  l.setStyleSheet(f'color:{col};font-size:12px;')
            return l

        nx0,ny0 = ls['nx0'],ls['ny0'];  nx1,ny1 = ls['nx1'],ls['ny1']
        if h == 'A_zigzag':  size_str = f'{nx0}x{ny0} Zigzag'
        elif h in CYL_TYPES: size_str = f'{nx0}x{ny0} Cylinder'
        else:                 size_str = f'{nx0}x{ny0} x {nx1}x{ny1}'

        il.addWidget(_ro(f'H type:  {h}', ACCENT),       0, 0)
        il.addWidget(_ro(f'Size:  {size_str}'),           0, 1)
        il.addWidget(_ro(f'Sites:  {ls["NL"]}'),          0, 2)
        il.addWidget(_ro(f'kin = {ls["kin"]:.4f}'),       1, 0)
        il.addWidget(_ro(f'kex = {ls["kex"]:.4f}'),       1, 1)
        il.addWidget(_ro(f'IN = {ls["isite"]}   OUT = {ls["osite"]}'), 1, 2)

        parts = []
        if h in ('IQH_IQH','IQH_AQH','IQH_cyl'):
            parts.append(f'phi_IQH_0 = {_phi_str(ls["phi_iqh0"])}')
        if h in ('IQH_IQH','AQH_IQH'):
            parts.append(f'phi_IQH_1 = {_phi_str(ls["phi_iqh1"])}')
        if h in ('AQH_AQH','AQH_IQH','A_zigzag','AQH_cyl'):
            parts.append(f'phi_AQH_0 = {_phi_str(ls["phi_aqh0"])}')
        if h in ('AQH_AQH','IQH_AQH'):
            parts.append(f'phi_AQH_1 = {_phi_str(ls["phi_aqh1"])}')
        if ls.get('is_cyl') and ls.get('psi', 0):
            parts.append(f'psi = {_phi_str(ls["psi"])}')
        il.addWidget(_ro('  |  '.join(parts) if parts else 'no phases'), 2, 0, 1, 3)

        def _collapsible(summary, detail, col):
            """Button that toggles a detail label — click to expand/collapse."""
            cw  = QWidget();  cl = QVBoxLayout(cw)
            cl.setContentsMargins(0,0,0,0);  cl.setSpacing(1)
            btn = QPushButton(f'▶  {summary}')
            btn.setCheckable(True)
            btn.setStyleSheet(
                f'QPushButton{{color:{col};background:transparent;border:none;'
                f'font-size:12px;text-align:left;padding:0px;}}'
                f'QPushButton:hover{{color:#ffffff;}}')
            lbl = QLabel(detail)
            lbl.setStyleSheet(f'color:{col};font-size:11px;padding-left:14px;')
            lbl.setWordWrap(True);  lbl.setVisible(False)
            def _toggle(checked, b=btn, l=lbl, s=summary):
                b.setText(('▼' if checked else '▶') + f'  {s}')
                l.setVisible(checked)
            btn.toggled.connect(_toggle)
            cl.addWidget(btn);  cl.addWidget(lbl)
            return cw

        heaters = ls.get('heaters', {})
        defects = ls.get('defects', set())
        if heaters:
            ht_detail = '\n'.join([f'site {k}: {v:+.2f} J' for k,v in sorted(heaters.items())])
            il.addWidget(_collapsible(f'Heaters ({len(heaters)})', ht_detail, '#ffcc00'), 3, 0, 1, 2)
        else:
            il.addWidget(_ro('Heaters:  none'), 3, 0)
        if defects:
            dt_detail = '\n'.join([f'site {k}' for k in sorted(defects)])
            il.addWidget(_collapsible(f'Defects zeroed ({len(defects)})', dt_detail, '#ff6b6b'), 3, 2)
        else:
            il.addWidget(_ro('Defects:  none'), 3, 2)

        gi.setMaximumWidth(500)
        row.addWidget(gi)

        # ── 2. Comb & Stepper ─────────────────────────────────────────
        ge = QGroupBox('Comb & Stepper');  ge.setFixedWidth(280)
        el = QGridLayout(ge);  el.setSpacing(4)
        self.spn_fsr   = QSpinBox();       self.spn_fsr.setRange(1,512);       self.spn_fsr.setValue(128)
        self.spn_D2    = QDoubleSpinBox(); self.spn_D2.setDecimals(4);         self.spn_D2.setRange(0,1);      self.spn_D2.setValue(0.004);  self.spn_D2.setSingleStep(0.001)
        self.spn_niter = QSpinBox();       self.spn_niter.setRange(100,50000); self.spn_niter.setValue(3000);  self.spn_niter.setSingleStep(500)
        self.spn_dt    = QDoubleSpinBox(); self.spn_dt.setDecimals(3);         self.spn_dt.setRange(0.001,1.0); self.spn_dt.setValue(0.1);   self.spn_dt.setSingleStep(0.01)
        for c,lbl,w_ in [(0,'+-FSR',self.spn_fsr),(1,'D2',self.spn_D2)]:
            el.addWidget(QLabel(lbl),0,c); el.addWidget(w_,1,c)
        for c,lbl,w_ in [(0,'NIter',self.spn_niter),(1,'dt/J',self.spn_dt)]:
            el.addWidget(QLabel(lbl),2,c); el.addWidget(w_,3,c)
        row.addWidget(ge)

        # ── 3. Sweep Schedule ─────────────────────────────────────────
        gf = QGroupBox('Sweep Schedule')
        gf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        fl = QVBoxLayout(gf);  fl.setSpacing(4)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel('N steps:'))
        self.spn_nsteps = QSpinBox();  self.spn_nsteps.setRange(2,10000); self.spn_nsteps.setValue(600)
        n_row.addWidget(self.spn_nsteps);  n_row.addStretch()
        fl.addLayout(n_row)

        sep2 = QFrame();  sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet('color:#1e2230;margin:2px 0;');  fl.addWidget(sep2)

        hdr = QHBoxLayout()
        for txt, w_ in [('Parameter', 155), ('Mode', 88), ('Value(s)', 210)]:
            l = QLabel(txt);  l.setFixedWidth(w_)
            l.setStyleSheet('color:#4a5270;font-size:11px;font-weight:bold;')
            hdr.addWidget(l)
        hdr.addStretch();  fl.addLayout(hdr)

        self._rows_container = QWidget()
        self._rows_vlay      = QVBoxLayout(self._rows_container)
        self._rows_vlay.setSpacing(2);  self._rows_vlay.setContentsMargins(0,0,0,0)
        self._rows_vlay.addStretch()

        scroll = QScrollArea();  scroll.setWidget(self._rows_container)
        scroll.setWidgetResizable(True);  scroll.setFixedHeight(110)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        fl.addWidget(scroll)

        # Always-present external rows
        self._add_param_row('Pump F',     'F',        0.08,  0.3,  10.0, removable=False)

        # Default detuning sweep range: inherit from Linear if available.
        # Linear convention: user specifies arbitrary start/end for their sweep
        # (commonly start=1.455 blue, end=1.4 red). Nonlinear wants a
        # redward sweep (decreasing detuning), so we force start > end
        # regardless of the order the user set in Linear.
        _ls = getattr(self, '_ls', {}) or {}
        _sw_a = _ls.get('sw_start', None)
        _sw_b = _ls.get('sw_end',   None)
        if _sw_a is not None and _sw_b is not None:
            _det_from = max(float(_sw_a), float(_sw_b))   # larger = start
            _det_to   = min(float(_sw_a), float(_sw_b))   # smaller = end
            # Fixed value sits midway (unused unless user switches to 'Fixed' mode)
            _det_fix  = 0.5 * (_det_from + _det_to)
        else:
            _det_fix, _det_from, _det_to = 1.455, 1.455, 1.4
        self._add_param_row('Detuning δ', 'detuning',
                            _det_fix, _det_from, _det_to, removable=False,
                            default_mode='Sweep')
        if self._is_cyl:
            self._add_param_row('Ext. flux Psi  (×π)', 'psi', 0.0, 0.0, 1.0, removable=False)
        # Inherited heater rows appended after _build_ui() returns (in __init__)

        self.spn_nsteps.valueChanged.connect(self._draw_sched_preview)
        row.addWidget(gf, stretch=1)
        gf.setMaximumWidth(480)

        # ── 4. Run ────────────────────────────────────────────────────
        gr  = QGroupBox('Run');  gr.setMinimumWidth(360);  gr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl2 = QGridLayout(gr);  rl2.setSpacing(4)
        self.spn_dynp = QSpinBox();  self.spn_dynp.setRange(1,200); self.spn_dynp.setValue(10)
        self.btn_run  = QPushButton('Run');      self.btn_run.setFixedHeight(30)
        self.btn_stop = QPushButton('Stop');     self.btn_stop.setFixedHeight(30)
        self.btn_save = QPushButton('💾  Save'); self.btn_save.setFixedHeight(30)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.prog_bar  = QProgressBar();  self.prog_bar.setRange(0,100); self.prog_bar.setValue(0)
        self.lbl_eta   = QLabel('');  self.lbl_eta.setStyleSheet('color:#4a5270;font-size:12px;')
        self.lbl_log   = QLabel('Ready');  self.lbl_log.setStyleSheet('color:#4a5270;font-size:12px;')
        self.sld_probe = QSlider(Qt.Horizontal);  self.sld_probe.setRange(0,299); self.sld_probe.setValue(0)
        self.lbl_probe = QLabel('Probe: step 0');  self.lbl_probe.setStyleSheet('color:#c8d0e7;font-size:12px;')
        self.spn_probe = QSpinBox()
        self.spn_probe.setRange(0, 299);  self.spn_probe.setValue(0)
        self.spn_probe.setFixedWidth(72)
        self.spn_probe.setStyleSheet('font-size:12px;')
        self.sld_probe.valueChanged.connect(self._on_probe_change)
        self.spn_probe.valueChanged.connect(self._on_probe_spinbox_change)
        lbl_dynp = QLabel('Plot every:');  lbl_dynp.setStyleSheet('color:#4a5270;font-size:12px;')
        for c in range(4): rl2.setColumnStretch(c, 1)
        # row 0: Plot every | spinbox | Save (right side)
        rl2.addWidget(lbl_dynp,      0,0); rl2.addWidget(self.spn_dynp, 0,1)
        rl2.addWidget(self.btn_save, 0,2,1,2)
        # row 1: Run and Stop take full width equally
        rl2.addWidget(self.btn_run,  1,0,1,2); rl2.addWidget(self.btn_stop,1,2,1,2)
        rl2.addWidget(self.prog_bar, 2,0,1,3);  rl2.addWidget(self.lbl_eta,2,3)
        rl2.addWidget(self.lbl_probe,3,0,1,2);  rl2.addWidget(self.sld_probe,3,2,1,1); rl2.addWidget(self.spn_probe,3,3)
        rl2.addWidget(self.lbl_log,  4,0,1,4)

        # ── Slow-time analysis (visible after run) ─────────────────
        sep3 = QFrame();  sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet('color:#1e2230;');
        rl2.addWidget(sep3, 5, 0, 1, 4)

        # parameter readout at selected step
        self.lbl_step_params = QLabel('—')
        self.lbl_step_params.setStyleSheet('color:#a8ff78;font-size:11px;')
        self.lbl_step_params.setWordWrap(True)
        rl2.addWidget(self.lbl_step_params, 6, 0, 1, 4)

        # slow time button
        self.btn_slow = QPushButton('▶  Run Slow Time Analysis')
        self.btn_slow.setFixedHeight(30)
        self.btn_slow.setEnabled(False)
        self.btn_slow.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_slow.clicked.connect(self._open_slow_time)
        rl2.addWidget(self.btn_slow, 8, 0, 1, 4)

        row.addWidget(gr, stretch=1)
        return w

    # ------------------------------------------------------------------
    # Parameter row management
    # ------------------------------------------------------------------
    def _add_param_row(self, label, pkey,
                       default_fixed=0.0, default_from=1.8, default_to=1.2,
                       removable=False, default_mode='Fixed'):
        pr = ParameterRow(label, pkey,
                          default_fixed=default_fixed,
                          default_from=default_from,
                          default_to=default_to,
                          removable=removable,
                          default_mode=default_mode,
                          get_nsteps=lambda: self.spn_nsteps.value())
        if removable:
            pr.btn_rm.clicked.connect(lambda: self._remove_param_row(pr, pkey))
        count = self._rows_vlay.count()
        self._rows_vlay.insertWidget(count - 1, pr)
        self._param_rows.append(pr)
        pr.mode_combo.currentTextChanged.connect(lambda _: self._draw_sched_preview())
        pr.spn_fixed.valueChanged.connect(lambda _: self._draw_sched_preview())
        pr.spn_from.valueChanged.connect( lambda _: self._draw_sched_preview())
        pr.spn_to.valueChanged.connect(   lambda _: self._draw_sched_preview())
        self._draw_sched_preview()
        return pr

    def _remove_param_row(self, pr, pkey):
        if pkey.startswith('heater_'):
            self._h_sweep.discard(int(pkey.split('_')[1]))
            self._draw_lattice()
        pr.setParent(None)
        if pr in self._param_rows:
            self._param_rows.remove(pr)
        self._draw_sched_preview()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_lattice(self):
        ls = self._ls
        NL = ls['NL'];  Nx = ls['Nxt'];  Ny = ls['Nyt']
        ax = self.ax_lat;  ax.cla();  ax.set_facecolor(PANEL_BG)
        ax.set_aspect('equal');  ax.set_xticks([]);  ax.set_yticks([])
        ax.set_title('Lattice power  --  click to add sweep channel',
                     color=TEXT_COL, fontsize=10, pad=3)
        for sp in ax.spines.values(): sp.set_edgecolor(SPINE_COL)
        if not ls.get('is_cyl'):
            ax.set_xlim(-0.8, Nx-0.2);  ax.set_ylim(-0.8, Ny-0.2)

        coords = {n: ((n-1)%Nx, (n-1)//Nx) for n in range(1, NL+1)}
        for n in range(1, NL+1):
            xn,yn = coords[n];  X=(n-1)%Nx;  Y=(n-1)//Nx
            if X < Nx-1:
                nb=n+1;  x2,y2=coords[nb]
                ax.plot([xn,x2],[yn,y2],color='#1e2a40',lw=0.9,zorder=1)
            if Y < Ny-1:
                nb=n+Nx;  x2,y2=coords[nb]
                ax.plot([xn,x2],[yn,y2],color='#1e2a40',lw=0.9,zorder=1)

        defects = ls.get('defects', set())
        for n in range(1, NL+1):
            xn,yn = coords[n]
            if n in defects:                    ec='#333355'
            elif n==ls['isite']:                ec='#4a9eff'
            elif n==ls['osite']:                ec='#ff4a6e'
            elif n in self._h_sweep:            ec='#ff8c42'
            elif ls.get('heaters',{}).get(n,0): ec='#ffcc00'
            else:                               ec='#2d4a6a'

            if self._results is not None and n not in defects:
                step = int(self.sld_probe.value())
                aW   = np.fft.fft(self._results['a_T'][:,:,step],axis=1,norm='ortho')
                pw   = np.sum(np.abs(aW)**2,axis=1)
                fc   = cm.hot(float(np.clip(pw[n-1]/(pw.max() or 1.),0,1)))
            elif n in defects:  fc = '#0a0a0f'
            else:               fc = '#0e1c2f'

            ax.add_patch(mpatches.Circle((xn,yn),RING_R,edgecolor=ec,facecolor=fc,linewidth=1.3,zorder=2))
            ax.add_patch(mpatches.Circle((xn,yn),RING_R*0.62,edgecolor='none',facecolor=DARK_BG,zorder=3))
            label,lc = None,ec
            if n in defects:                        label,lc='×','#555577'
            elif n==ls['isite']:                    label,lc='IN','#4a9eff'
            elif n==ls['osite']:                    label,lc='OUT','#ff4a6e'
            elif n in self._h_sweep:                label,lc='~','#ff8c42'
            elif ls.get('heaters',{}).get(n,0):     label,lc=f"{ls['heaters'][n]:+.1f}",'#ffcc00'
            if label:
                ax.text(xn,yn,label,ha='center',va='center',
                        fontsize=6,color=lc,fontweight='bold',zorder=4)
        self.canvas_lat.draw_idle()

    def _draw_sched_preview(self, *_):
        n = max(2, self.spn_nsteps.value());  x = np.arange(n)

        # Categorise rows
        f_rows   = [pr for pr in self._param_rows if pr.pkey == 'F']
        det_rows = [pr for pr in self._param_rows if pr.pkey == 'detuning']
        psi_rows = [pr for pr in self._param_rows if pr.pkey == 'psi']
        h_rows   = [pr for pr in self._param_rows if pr.pkey.startswith('heater_')]
        has_psi  = len(psi_rows) > 0
        has_h    = len(h_rows)   > 0

        # Build gridspec dynamically
        n_plots = 1 + (1 if has_psi else 0) + (1 if has_h else 0)
        self.fig_sched.clear()
        gs = self.fig_sched.add_gridspec(n_plots, 1, hspace=0.55,
             left=0.13, right=0.87, top=0.95, bottom=0.06)

        def _style(ax, title, col):
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, color=col, fontsize=9, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=8)

        # ── Subplot 0: Pump F (left) + Detuning δ (right) twin axis ──
        ax_f = self.fig_sched.add_subplot(gs[0])
        ax_d = ax_f.twinx()
        _style(ax_f, 'Pump F  &  Detuning δ', TEXT_COL)

        if f_rows:
            yf = f_rows[0].get_array(n)
            ax_f.plot(x, yf, color='#ffcc00', lw=1.6,
                      ls='-' if f_rows[0].is_active() else '--')
        ax_f.set_ylabel('F', color='#ffcc00', fontsize=9)
        ax_f.tick_params(axis='y', labelcolor='#ffcc00', labelsize=8)
        ax_f.yaxis.label.set_color('#ffcc00')

        if det_rows:
            yd = det_rows[0].get_array(n)
            ax_d.plot(x, yd, color='#4a9eff', lw=1.6,
                      ls='-' if det_rows[0].is_active() else '--')
        ax_d.set_ylabel('Δ/J', color='#4a9eff', fontsize=9)
        ax_d.tick_params(axis='y', labelcolor='#4a9eff', labelsize=8)
        ax_d.yaxis.label.set_color('#4a9eff')
        # style twinx spines
        for sp in ax_d.spines.values(): sp.set_edgecolor('#3a4560')

        plot_idx = 1

        # ── Subplot 1 (optional): External flux Psi ───────────────────
        if has_psi:
            ax_psi = self.fig_sched.add_subplot(gs[plot_idx])
            _style(ax_psi, 'Ext. flux  Psi', '#ff8c42')
            yp = psi_rows[0].get_array(n)
            ax_psi.plot(x, yp, color='#ff8c42', lw=1.6,
                        ls='-' if psi_rows[0].is_active() else '--')
            ax_psi.set_ylabel('Psi (×π)', color='#ff8c42', fontsize=9)
            ax_psi.tick_params(axis='y', labelcolor='#ff8c42', labelsize=8)
            plot_idx += 1

        # ── Subplot 2 (optional): All heater channels ─────────────────
        if has_h:
            ax_h = self.fig_sched.add_subplot(gs[plot_idx])
            _style(ax_h, 'Heaters', '#a8ff78')
            COLS = ['#a8ff78','#ffcc00','#ff4a6e','#cc88ff','#ff8c42','#4a9eff']
            for ci, pr in enumerate(h_rows):
                yh = pr.get_array(n)
                site = pr.pkey.split('_')[1]
                ax_h.plot(x, yh, color=COLS[ci%len(COLS)], lw=1.4,
                          ls='-' if pr.is_active() else '--',
                          label=f'H{site}')
            ax_h.set_ylabel('delta (J)', color='#a8ff78', fontsize=9)
            ax_h.tick_params(axis='y', labelcolor='#a8ff78', labelsize=8)
            ax_h.legend(fontsize=7, frameon=False, labelcolor=TEXT_COL,
                        ncol=min(len(h_rows), 4), loc='best')

        # ── Real-time step indicator (moves during simulation) ───────
        n_full = max(2, self.spn_nsteps.value())
        self._sched_vline     = ax_f.axvline(x=0, color='white', lw=1.2, ls=':', alpha=0.85)
        self._sched_vline_psi = ax_psi.axvline(x=0, color='white', lw=1.2, ls=':', alpha=0.85) if has_psi else None
        self._sched_vline_h   = ax_h.axvline(  x=0, color='white', lw=1.2, ls=':', alpha=0.85) if has_h   else None
        ax_f.set_xlim(0, n_full - 1)

        self.canvas_sched.draw_idle()

    def _init_spec_plots(self, x_label):
        for ax,col,ttl in [
            (self.ax_pump,    '#4a9eff','Pump power  (output ring)'),
            (self.ax_side,    '#ff4a6e','Comb power excl. pump  (output ring)'),
            (self.ax_pump_all,'#4a9eff','Pump power  (all rings)'),
            (self.ax_side_all,'#ff4a6e','Comb power excl. pump  (all rings)')]:
            ax.cla();  ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl,color=col,fontsize=8,pad=3)
            ax.set_xlabel(x_label,color=TEXT_DIM,fontsize=8)
            ax.tick_params(colors=TEXT_COL,labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True,color=GRID_COL,lw=0.4)
        self._line_pump,    =self.ax_pump.plot([],[],    color='#4a9eff',lw=1.5)
        self._line_side,    =self.ax_side.plot([],[],    color='#ff4a6e',lw=1.5)
        self._line_pump_all,=self.ax_pump_all.plot([],[], color='#4a9eff',lw=1.5)
        self._line_side_all,=self.ax_side_all.plot([],[], color='#ff4a6e',lw=1.5)
        self._vline_pump    =self.ax_pump.axvline(0,    color='white',lw=1,ls='--',alpha=0.5)
        self._vline_side    =self.ax_side.axvline(0,    color='white',lw=1,ls='--',alpha=0.5)
        self._vline_pump_all=self.ax_pump_all.axvline(0,color='white',lw=1,ls='--',alpha=0.5)
        self._vline_side_all=self.ax_side_all.axvline(0,color='white',lw=1,ls='--',alpha=0.5)
        self.canvas_plots.draw_idle()

    def _update_spec_plots(self, n_done, probe_i):
        xsf = self._xv[:n_done]
        for arr, line, ax, vl in [
            (self._pump_n,     self._line_pump,     self.ax_pump,     self._vline_pump),
            (self._side_n,     self._line_side,     self.ax_side,     self._vline_side),
            (self._pump_n_all, self._line_pump_all, self.ax_pump_all, self._vline_pump_all),
            (self._side_n_all, self._line_side_all, self.ax_side_all, self._vline_side_all),
        ]:
            data = arr[:n_done]
            mx   = data.max() or 1.
            line.set_data(xsf, data / mx)
            ax.set_xlim(xsf[0], xsf[-1]);  ax.set_ylim(0, 1.05)
            if n_done > probe_i: vl.set_xdata([xsf[probe_i], xsf[probe_i]])
        self.canvas_plots.draw_idle()

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------
    def _on_lat_click(self, event):
        if event.inaxes!=self.ax_lat or event.xdata is None: return
        ls=self._ls;  NL=ls['NL'];  Nx=ls['Nxt']
        best_n,best_d=None,float('inf')
        for n in range(1, NL+1):
            X=(n-1)%Nx;  Y=(n-1)//Nx
            d=(event.xdata-X)**2+(event.ydata-Y)**2
            if d<best_d: best_d=d; best_n=n
        if best_d>(RING_R*3.5)**2: return
        # clicking any ring adds it as a sweep channel (if not already present)
        pkey=f'heater_{best_n}'
        existing_pkeys = [pr.pkey for pr in self._param_rows]
        if pkey not in existing_pkeys:
            self._h_sweep.add(best_n)
            self._add_param_row(f'Heater {best_n}', pkey,
                                default_fixed=0.0,
                                default_from=0.0, default_to=3.0,
                                removable=True)
            self.status.showMessage(f'Sweep channel added: {pkey}')
            self._draw_lattice()

    def _on_spec_press(self,e): self._spec_drag=True;  self._probe_at(e)
    def _on_spec_move(self,e):
        if self._spec_drag: self._probe_at(e)

    def _probe_at(self, event):
        if self._xv is None: return
        # Accept clicks on any of the 4 sweep-power panels:
        # left column (output ring):  ax_pump, ax_side
        # right column (all rings):   ax_pump_all, ax_side_all
        if event.inaxes not in (self.ax_pump,     self.ax_side,
                                self.ax_pump_all, self.ax_side_all): return
        if event.xdata is None: return
        idx=int(np.argmin(np.abs(self._xv-event.xdata)))
        self.sld_probe.blockSignals(True);  self.sld_probe.setValue(idx)
        self.sld_probe.blockSignals(False);  self._on_probe_change(idx)

    def _on_probe_change(self, idx):
        self.lbl_probe.setText(f'Probe: step {idx}')
        # sync spinbox without re-triggering
        self.spn_probe.blockSignals(True);  self.spn_probe.setValue(idx);  self.spn_probe.blockSignals(False)
        if self._results is not None:
            self._draw_lattice()
            self._update_cross_sections(idx)
            self._update_heatmap_vlines(idx)
        if self._xv is not None and idx < len(self._xv):
            xp = self._xv[idx]
            for vl in (self._vline_pump, self._vline_side,
                       self._vline_pump_all, self._vline_side_all):
                if vl is not None: vl.set_xdata([xp, xp])
            self.canvas_plots.draw_idle()
        # move schedule preview vlines to current probe step
        if getattr(self, '_sched_vline', None) is not None:
            self._sched_vline.set_xdata([idx, idx])
        if getattr(self, '_sched_vline_psi', None) is not None:
            self._sched_vline_psi.set_xdata([idx, idx])
        if getattr(self, '_sched_vline_h', None) is not None:
            self._sched_vline_h.set_xdata([idx, idx])
        self.canvas_sched.draw_idle()
        # update parameter readout
        self._update_step_params(idx)

    def _on_probe_spinbox_change(self, val):
        self.sld_probe.blockSignals(True);  self.sld_probe.setValue(val);  self.sld_probe.blockSignals(False)
        self._on_probe_change(val)

    def _update_step_params(self, idx):
        """Show schedule parameter values at the selected step."""
        if self._results is None or not hasattr(self, '_sched_snapshot'):
            self.lbl_step_params.setText('—')
            return
        sched = self._sched_snapshot
        parts = []
        for pr in self._param_rows:
            arr = pr.get_array(sched.n_steps)
            if pr.pkey == 'psi':
                val = arr[idx] / np.pi
                parts.append(f'Psi={val:.4f}π')
            elif pr.pkey == 'F':
                parts.append(f'F={arr[idx]:.4f}')
            elif pr.pkey == 'detuning':
                parts.append(f'Δ={arr[idx]:.6f}')
            elif pr.pkey.startswith('heater_'):
                site = pr.pkey.split('_')[1]
                parts.append(f'H{site}={arr[idx]:.3f}J')
        self.lbl_step_params.setText('  '.join(parts) if parts else '—')

    def _update_cross_sections(self, idx):
        """Plot |a_W|² and |a_T|² at the selected step for osite."""
        if self._results is None: return
        a_T     = self._results['a_T']          # (NLattice, NumFSRs, N_steps)
        FSRVec  = self._results['FSRVec']       # FFT-ordered [0,1,...,+N,-N,...,-1]
        osite_i = self._ls['osite'] - 1
        a_T_step = a_T[osite_i, :, idx]        # (NumFSRs,)
        a_W_step = np.fft.fft(a_T_step, norm='ortho')   # pump at index 0
        pW = np.abs(a_W_step)**2
        pT = np.abs(a_T_step)**2

        # Shift to centered μ ordering [-N,...,0,...,+N] for a natural plot
        mu_centered  = np.fft.fftshift(FSRVec)
        pW_centered  = np.fft.fftshift(pW)

        ax = self.ax_xsec_aW;  ax.cla();  ax.set_facecolor(PANEL_BG)
        pW_log = 10 * np.log10(pW_centered + 1e-200)
        ax.plot(mu_centered, pW_log, color='#ff8c42', lw=1.2)
        ax.set_title(f'10·log|a_W|²  step {idx}  (osite)', color='#ff8c42', fontsize=8, pad=3)
        ax.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=7)
        ax.tick_params(colors=TEXT_COL, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4)
        fsr = len(FSRVec) // 2
        ax.set_xticks([-fsr, 0, fsr]); ax.set_xticklabels([f'{-fsr}', '0', f'{fsr}'], fontsize=7)

        ax = self.ax_xsec_aT;  ax.cla();  ax.set_facecolor(PANEL_BG)
        ax.plot(pT, color='#a8ff78', lw=1.2)
        ax.set_title(f'|a_T|²  step {idx}  (osite)', color='#a8ff78', fontsize=8, pad=3)
        ax.set_xlabel('θ index', color=TEXT_DIM, fontsize=7)
        ax.tick_params(colors=TEXT_COL, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4)
        ax.set_xticks([0, len(pT)-1]); ax.set_xticklabels(['0', '2π'], fontsize=7)

        self.canvas_xsec.draw_idle()

    def _update_heatmap_vlines(self, idx):
        """Overlay a vertical dashed line at the selected step on all heatmaps."""
        for key, vl_attr in [('aT_out','_hvl_aT_out'),('aT_all','_hvl_aT_all'),
                              ('aW_out','_hvl_aW_out'),('aW_all','_hvl_aW_all')]:
            ax = {'aT_out':self.ax_aT_out,'aT_all':self.ax_aT_all,
                  'aW_out':self.ax_aW_out,'aW_all':self.ax_aW_all}[key]
            vl = getattr(self, vl_attr, None)
            if vl is None:
                vl = ax.axvline(idx, color='cyan', lw=1, ls='--', alpha=0.7)
                setattr(self, vl_attr, vl)
            else:
                vl.set_xdata([idx, idx])
        self.canvas_heat.draw_idle()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self):
        n  = self.spn_nsteps.value()
        dp = self.spn_dynp.value()
        ls = self._ls
        self._nl_session_folder = None   # reset so folder name reflects new params

        sched = Schedule(n)
        for pr in self._param_rows:
            arr = pr.get_array(n)
            if pr.pkey == 'psi':
                arr = arr * np.pi   # user enters in units of π
            sched.add(pr.pkey, arr)

        x_lbl        = 'Step'
        primary_pkey = next((pr.pkey for pr in self._param_rows if pr.is_active()), 'detuning')

        # Inherited heaters are baked into H_mat by Linear.py.
        # Subtract them so SweepRunner can re-apply cleanly via
        # the heater_N schedule channels without double-counting.
        H_clean = ls['H_mat'][1:,1:].copy()
        for site, delta in ls.get('heaters', {}).items():
            if delta != 0:
                H_clean[site - 1, site - 1] -= delta

        fp = dict(
            H_mat        = H_clean,
            NLattice     = ls['NL'],
            ISite        = ls['isite'],  OSite = ls['osite'],
            kin          = ls['kin'],    kex   = ls['kex'],
            one_side_fsr = self.spn_fsr.value(),
            D2           = self.spn_D2.value(),
            NIter        = self.spn_niter.value(),
            TStep        = self.spn_dt.value(),
            # lattice params needed for ψ H rebuild
            h_str        = ls['h_str'],
            nx0          = ls['nx0'],    ny0   = ls['ny0'],
            nx1          = ls['nx1'],    ny1   = ls['ny1'],
            j1           = ls.get('j1', 0.3),
            phi_iqh0     = ls.get('phi_iqh0', np.pi/2),
            phi_iqh1     = ls.get('phi_iqh1', np.pi/2),
            phi_aqh0     = ls.get('phi_aqh0', np.pi/4),
            phi_aqh1     = ls.get('phi_aqh1', np.pi/4),
        )

        NumFSRs = 2*fp['one_side_fsr']+1
        FSRVec  = np.round(np.fft.fftfreq(NumFSRs, d=1.0) * NumFSRs).astype(float)  # FFT-order
        PumpFSR = 0;  OSite_i=fp['OSite'];  steps=n

        self._xv          = np.arange(steps, dtype=float)
        self._pump_n      = np.zeros(steps);  self._side_n     = np.zeros(steps)
        self._pump_n_all  = np.zeros(steps);  self._side_n_all = np.zeros(steps)
        self._FSRVec      = FSRVec;  self._PumpFSR=PumpFSR;  self._n_steps=steps

        self._sched_snapshot = sched   # keep for parameter readout at any step
        self._run_params_snapshot = dict(   # scalar params at launch time
            one_side_fsr = fp['one_side_fsr'],
            D2           = fp['D2'],
            NIter        = fp['NIter'],
            TStep        = fp['TStep'],
        )
        self._init_spec_plots(x_lbl)
        self.sld_probe.setRange(0,steps-1);  self.sld_probe.setValue(0)
        self.spn_probe.setRange(0,steps-1);  self.spn_probe.setValue(0)
        self.prog_bar.setValue(0)
        self.btn_run.setEnabled(False);  self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False);  self.btn_slow.setEnabled(False)
        self.btn_slow.setEnabled(False)
        self.lbl_log.setText('Running...');  self._stop_flag.clear()

        runner=SweepRunner(fp);  _q=self._q

        def worker():
            import time
            def cb(gg,aT,sp,elapsed):
                # output ring
                aW  = np.fft.fft(aT[OSite_i-1,:],norm='ortho');  P  = np.abs(aW)**2
                pv  = P[PumpFSR]
                mask= np.ones(NumFSRs,bool);  mask[PumpFSR]=False
                sv  = P[mask].sum()
                # all rings summed
                aW_all = np.fft.fft(aT,axis=1,norm='ortho')  # (NLattice, NumFSRs)
                P_all  = np.abs(aW_all)**2
                Psum   = P_all.sum(axis=0)
                pv_all = Psum[PumpFSR]
                sv_all = Psum[mask].sum()
                em,es=divmod(int(elapsed/(gg+1)*(steps-gg-1)) if gg>0 else 0,60)
                _q.put(('step',gg,pv,sv,pv_all,sv_all,em,es))
                if self._stop_flag.is_set(): raise StopIteration
            try:
                res=runner.run(sched,callback=cb)
                if self._stop_flag.is_set():
                    _q.put(('stopped', res))
                else:
                    _q.put(('done', res))
            except Exception as exc:
                _q.put(('error',str(exc)));  import traceback; traceback.print_exc()

        threading.Thread(target=worker,daemon=True).start()
        self._poll_timer.start()

    def _poll_queue(self):
        dp=self.spn_dynp.value();  n=self._n_steps
        try:
            while True:
                msg=self._q.get_nowait(); kind=msg[0]
                if kind=='step':
                    _,gg,pv,sv,pv_all,sv_all,em,es=msg
                    self._pump_n[gg]=pv;      self._side_n[gg]=sv
                    self._pump_n_all[gg]=pv_all; self._side_n_all[gg]=sv_all
                    nd=gg+1
                    # advance the schedule vlines
                    if getattr(self, '_sched_vline', None) is not None:
                        self._sched_vline.set_xdata([gg, gg])
                    if getattr(self, '_sched_vline_psi', None) is not None:
                        self._sched_vline_psi.set_xdata([gg, gg])
                    if getattr(self, '_sched_vline_h', None) is not None:
                        self._sched_vline_h.set_xdata([gg, gg])
                    self.canvas_sched.draw_idle()
                    if gg%dp==0 or gg==n-1:
                        self._update_spec_plots(nd,int(self.sld_probe.value()))
                        self._draw_lattice()
                    self.prog_bar.setValue(int(100*nd/n))
                    self.lbl_eta.setText(f'{em}m{es:02d}s left')
                    self.status.showMessage(f'Step {nd}/{n}  |  ETA {em}m{es:02d}s')
                elif kind=='done':    self._on_sweep_done(msg[1])
                elif kind=='stopped': self._on_sweep_stopped(msg[1])
                elif kind=='error':   self._on_sweep_finished(f'Error: {msg[1]}')
        except queue.Empty:
            pass

    def _on_sweep_done(self, res):
        self._results = res
        self._update_spec_plots(self._n_steps, int(self.sld_probe.value()))
        self._draw_lattice()
        self._draw_heat_maps()
        self._update_cross_sections(int(self.sld_probe.value()))
        self._update_step_params(int(self.sld_probe.value()))
        self.btn_save.setEnabled(True)
        self._on_sweep_finished('Done.')

    def _on_sweep_stopped(self, res):
        """Stop was clicked — store and display whatever steps completed."""
        if res is not None and res.get('n_done', 0) > 0:
            self._results = res
            n_done = res['n_done']
            probe  = min(int(self.sld_probe.value()), n_done - 1)
            self._update_spec_plots(n_done, probe)
            self._draw_lattice()
            self._draw_heat_maps()
            self._update_cross_sections(probe)
            self._update_step_params(probe)
            self.btn_save.setEnabled(True)
            self._on_sweep_finished(f'Stopped — {n_done} steps saved.')
        else:
            self._on_sweep_finished('Stopped.')

    def _draw_heat_maps(self):
        if self._results is None: return
        a_T = self._results['a_T']          # (NLattice, NumFSRs, N_steps) — θ-space
        a_W = np.fft.fft(a_T, axis=1, norm='ortho')   # FFT-order: index 0 = μ=0
        OSite_i = self._ls['osite'] - 1    # 0-indexed

        # fftshift the μ-axis of aW to center μ=0 for display. aT is in θ and
        # does not need shifting (θ axis is just [0, 2π)).
        aW_out = np.fft.fftshift(np.abs(a_W[OSite_i, :, :])**2,          axes=0)
        aW_all = np.fft.fftshift((np.abs(a_W)**2).sum(axis=0),           axes=0)

        datasets = {
            'aT_out': np.abs(a_T[OSite_i, :, :])**2,          # (NumFSRs, N_steps)
            'aT_all': (np.abs(a_T)**2).sum(axis=0),            # sum of intensities over rings
            'aW_out': aW_out,
            'aW_all': aW_all,
        }

        axes = {
            'aT_out': self.ax_aT_out,
            'aT_all': self.ax_aT_all,
            'aW_out': self.ax_aW_out,
            'aW_all': self.ax_aW_all,
        }
        ylabels = {
            'aT_out': 'θ index', 'aT_all': 'θ index',
            'aW_out': 'FSR  μ',  'aW_all': 'FSR  μ',
        }
        titles = {
            'aT_out': ('|a_T|²  (output ring)', '#a8ff78'),
            'aT_all': ('|a_T|²  (all rings)',    '#a8ff78'),
            'aW_out': ('|a_W|²  (output ring)',  '#ff8c42'),
            'aW_all': ('|a_W|²  (all rings)',    '#ff8c42'),
        }
        FSRVec = self._results['FSRVec']
        steps  = a_T.shape[2]
        NumFSRs = a_T.shape[1]
        one_side = NumFSRs // 2

        for key, data in datasets.items():
            ax  = axes[key]
            ttl, col = titles[key]
            # aW data has already been fftshifted above, so the centered μ range is [-one_side, +one_side]
            ymin = -one_side if 'aW' in key else 0
            ymax = +one_side if 'aW' in key else NumFSRs - 1

            if 'aW' in key:
                # log scale: add tiny floor before log to avoid -inf
                data_plot = 10 * np.log10(data + 1e-200)
                vmin_plot = data_plot.max() - 60   # show 60 dB of dynamic range
                vmax_plot = data_plot.max() or 0.
                cmap_key  = 'hot'
            else:
                # linear, normalised 0-1
                mx = data.max() or 1.
                data_plot = data / mx
                vmin_plot, vmax_plot = 0, 1
                cmap_key  = 'viridis'

            if key not in self._heat_imgs:
                img = ax.imshow(data_plot, aspect='auto', origin='lower',
                                extent=[0, steps-1, ymin, ymax],
                                cmap=cmap_key, vmin=vmin_plot, vmax=vmax_plot,
                                interpolation='nearest')
                self._heat_imgs[key] = img
                ax.set_xlim(0, steps-1);  ax.set_ylim(ymin, ymax)
            else:
                self._heat_imgs[key].set_data(data_plot)
                self._heat_imgs[key].set_extent([0, steps-1, ymin, ymax])
                self._heat_imgs[key].set_clim(vmin_plot, vmax_plot)
                self._heat_imgs[key].set_cmap(cmap_key)

            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(ylabels[key], color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            # x-axis: leftmost and rightmost only
            ax.set_xticks([0, steps-1])
            ax.set_xticklabels(['0', f'{steps-1}'], fontsize=7)
            if 'aW' in key:
                one_side = NumFSRs // 2
                ax.set_yticks([-one_side, 0, one_side])
                ax.set_yticklabels([f'{-one_side}', '0', f'{one_side}'], fontsize=7)
            else:
                ax.set_yticks([0, NumFSRs - 1])
                ax.set_yticklabels(['0', '2π'], fontsize=7)

        self.canvas_heat.draw_idle()

    def _on_sweep_finished(self, msg):
        self._poll_timer.stop()
        self.btn_run.setEnabled(True);  self.btn_stop.setEnabled(False)
        self.lbl_log.setText(msg);  self.status.showMessage(msg)

    def _open_slow_time(self):
        if self._results is None or not hasattr(self, '_sched_snapshot'):
            self.status.showMessage('No simulation data — run first.'); return
        idx      = int(self.sld_probe.value())
        a_T_step = self._results['a_T'][:, :, idx].copy()
        niter    = 100000  # NIter for slow-time run; adjustable inside SlowTimeWindow
        sched    = self._sched_snapshot
        sp       = sched.at(idx)

        # Inherited heaters already baked into H_mat — subtract them so
        # SlowTimeWindow can apply heater_N channels cleanly
        H_clean = self._ls['H_mat'][1:,1:].copy()
        for site, delta in self._ls.get('heaters', {}).items():
            if delta != 0:
                H_clean[site - 1, site - 1] -= delta

        # Create step subfolder inside the NL session folder
        nl_folder  = self._get_nl_folder()
        slow_folder = os.path.join(nl_folder, f'step_{idx}')
        os.makedirs(slow_folder, exist_ok=True)

        params = dict(
            a_T_init     = a_T_step,
            step_idx     = idx,
            NIter_more   = niter,
            detuning     = sp.get('detuning', 1.5),
            F            = sp.get('F',        0.3),
            psi          = sp.get('psi',      0.0),
            heaters      = {k: v for k, v in sp.items() if k.startswith('heater_')},
            H_mat        = H_clean,
            NLattice     = self._ls['NL'],
            ISite        = self._ls['isite'],
            OSite        = self._ls['osite'],
            kin          = self._ls['kin'],
            kex          = self._ls['kex'],
            one_side_fsr = self.spn_fsr.value(),
            D2           = self.spn_D2.value(),
            TStep        = self.spn_dt.value(),
            FSRVec       = self._results['FSRVec'],
            # full schedule channels — every parameter (Δ, F, ψ, heater_N…)
            # as a complete array so stability analysis can use correct values
            # at any chosen step, not just the launch step
            schedule     = {k: v.copy() for k, v in sched.channels.items()},
            # sweep progress power curves for the static indicator panel
            sweep_pump_out  = self._pump_n.copy()     if self._pump_n     is not None else None,
            sweep_side_out  = self._side_n.copy()     if self._side_n     is not None else None,
            sweep_pump_all  = self._pump_n_all.copy() if self._pump_n_all is not None else None,
            sweep_side_all  = self._side_n_all.copy() if self._side_n_all is not None else None,
            # save path for slow time window
            session_folder  = slow_folder,
            # full linear state for geometry/visualization
            linear_state = self._ls,
        )
        self._slow_win = SlowTimeWindow(params, parent=self)
        self._slow_win.show()

    def _on_stop(self):
        self._stop_flag.set();  self.lbl_log.setText('Stopping...')

    def _get_nl_folder(self):
        """Compute (and create) the NL session folder, caching it in self._nl_session_folder."""
        if getattr(self, '_nl_session_folder', None):
            return self._nl_session_folder

        base  = self._ls.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
        _flt  = lambda v: f'{v:.4g}'.replace('.', 'p')

        n_planned = self.spn_nsteps.value()
        n_done    = (self._results.get('n_done', n_planned)
                     if self._results is not None else n_planned)
        stopped_early = (self._results is not None and n_done < n_planned)
        sched_obj     = (self._results.get('schedule') if self._results else None)

        def _param_str(pkey, prefix):
            rows = [pr for pr in self._param_rows if pr.pkey == pkey]
            if not rows: return ''
            pr   = rows[0]
            mode = pr.mode_combo.currentText()
            if mode == 'Fixed':
                return f"{prefix}{_flt(pr.spn_fixed.value())}"
            elif mode == 'Sweep':
                v_from = pr.spn_from.value()
                v_to   = pr.spn_to.value()
                if (stopped_early and sched_obj is not None
                        and pkey in sched_obj.channels and n_done > 0):
                    v_to = float(sched_obj.channels[pkey][n_done - 1])
                return f"{prefix}{_flt(v_from)}to{_flt(v_to)}"
            else:
                arr = pr.get_array(n_planned)
                return f"{prefix}{_flt(float(arr.min()))}to{_flt(float(arr.max()))}"

        f_str   = _param_str('F',        'F')
        d_str   = _param_str('detuning', 'δ')
        psi_str = _param_str('psi',      'psi')
        extras  = '_'.join(s for s in [f_str, d_str, psi_str] if s)

        name = (f"FSR{self.spn_fsr.value()}"
                f"_D2_{_flt(self.spn_D2.value())}"
                f"_NIter{self.spn_niter.value()}"
                f"_dt{_flt(self.spn_dt.value())}"
                f"_Nsteps{n_done}"
                + (f"_{extras}" if extras else ''))

        folder = os.path.join(base, name)
        os.makedirs(folder, exist_ok=True)
        self._nl_session_folder = folder
        return folder

    def _on_save(self):
        # ── Protection: disable all action buttons + show status while saving.
        # Saving can take seconds for a 600-step sweep (tens of MB of complex128
        # a_T array). processEvents() forces the UI to repaint the disabled
        # state before the blocking save starts.
        _prev_run_enabled  = self.btn_run.isEnabled()
        _prev_stop_enabled = self.btn_stop.isEnabled()
        _prev_save_enabled = self.btn_save.isEnabled()
        _prev_slow_enabled = self.btn_slow.isEnabled()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_slow.setEnabled(False)
        self.lbl_log.setText('💾  Saving... (please wait)')
        self.lbl_log.setStyleSheet('color:#ffcc00;font-size:12px;font-weight:bold;')
        QApplication.processEvents()

        try:
            folder = self._get_nl_folder()

            # ── Figures ───────────────────────────────────────────────────────
            n_planned = self.spn_nsteps.value()
            n_done    = (self._results.get('n_done', n_planned)
                         if self._results is not None else n_planned)
            # ── Figures: schedule, sweep power, heatmaps, cross-sections ──────
            for fig_, fname in [
                (self.fig_sched, 'schedule'),
                (self.fig_plots, 'sweep_power'),
                (self.fig_heat,  'heatmaps'),
                (self.fig_xsec,  'cross_sections'),
            ]:
                for fmt in ('png', 'svg'):
                    buf = io.BytesIO()
                    fig_.savefig(buf, format=fmt, dpi=200, bbox_inches='tight',
                                 facecolor=fig_.get_facecolor())
                    with open(os.path.join(folder, f'{fname}.{fmt}'), 'wb') as fh:
                        fh.write(buf.getvalue())

            # ── params.npz ────────────────────────────────────────────────────
            ls  = self._ls

            # Use the snapshot captured at run-time, not the current UI values
            # (user may have changed spinboxes since the simulation ran)
            sched_snap = getattr(self, '_sched_snapshot', None)
            def _sched_arr(key, n):
                if sched_snap is not None and key in sched_snap.channels:
                    return sched_snap.channels[key][:n]
                # fallback: reconstruct from UI (only if no snapshot available)
                arr = next((pr.get_array(n_planned) for pr in self._param_rows
                            if pr.pkey == key), np.array([]))
                return arr[:n]

            rp = getattr(self, '_run_params_snapshot', None)
            def _rp(key, fallback_widget):
                return rp[key] if rp and key in rp else fallback_widget()

            params = dict(
                h_str        = np.array([ls['h_str']]),
                kin          = np.array([ls['kin']]),
                kex          = np.array([ls['kex']]),
                ISite        = np.array([ls['isite']]),
                OSite        = np.array([ls['osite']]),
                one_side_fsr = np.array([_rp('one_side_fsr', self.spn_fsr.value)]),
                D2           = np.array([_rp('D2',           self.spn_D2.value)]),
                NIter        = np.array([_rp('NIter',        self.spn_niter.value)]),
                TStep        = np.array([_rp('TStep',        self.spn_dt.value)]),
                N_steps      = np.array([n_done]),
                F_schedule   = _sched_arr('F',        n_done),
                D_schedule   = _sched_arr('detuning', n_done),
            )
            # sched.channels['psi'] already stores radians (see _on_run: arr = arr * np.pi
            # converts user's "units of π" input to radians before Schedule.add).
            # Saving psi_arr directly matches the load path in Linear.py, which divides
            # by π to recover the spinbox value in "units of π".
            psi_arr = _sched_arr('psi', n_done)
            if len(psi_arr):
                params['psi_schedule'] = psi_arr
            if self._results is not None:
                params['FSRVec']  = self._results['FSRVec']
                params['PumpFSR'] = np.array([self._results['PumpFSR']])
                params['a_T']     = self._results['a_T']
            np.savez(os.path.join(folder, 'nonlinear_params.npz'), **params)

            # Success
            self.status.showMessage(f'✅ Saved → {folder}')
            self.lbl_log.setText(f'✅  Saved {os.path.basename(folder)}')
            self.lbl_log.setStyleSheet('color:#a8ff78;font-size:12px;')

        except Exception as exc:
            self.lbl_log.setText(f'❌  Save failed: {exc}')
            self.lbl_log.setStyleSheet('color:#ff4a6e;font-size:12px;font-weight:bold;')
            import traceback; traceback.print_exc()

        finally:
            # Restore button states. btn_slow should be enabled after a successful
            # save (it used to be), so we explicitly enable it here unless an
            # earlier error left btn_save disabled deliberately.
            self.btn_run.setEnabled(_prev_run_enabled)
            self.btn_stop.setEnabled(_prev_stop_enabled)
            self.btn_save.setEnabled(_prev_save_enabled)
            self.btn_slow.setEnabled(True)