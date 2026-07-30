"""Physical-consistency & spectral/gradient metrics for 2D Navier-Stokes (vorticity form).

Vendored from the rebuttal notes (``ns_spectral_gradient_metrics.py`` +
``physical_consistency_metrics.md``). The field is the scalar vorticity ``w`` on a periodic
unit torus ``[0, L]^2`` sampled on an ``N x N`` grid (default ``L=1``); velocity is
reconstructed spectrally from ``-Δψ = w``.

Two metric families, all comparing a prediction ``w_pred`` against ground truth ``w_true``
for a single (H, W) snapshot; average over batch/time in the caller:

* spectral / gradient resolution — is high-frequency energy / sharp structure preserved
  (``spectral bias``): energy-spectrum band errors, spectral centroid, enstrophy, gradient
  relative error, peak-gradient / P99 / total-variation ratios.
* conservation — do discrete invariants hold: total vorticity (mass), momentum, divergence
  of the reconstructed velocity (incompressibility ∇·u=0), kinetic energy, enstrophy.

All functions are NumPy; fields are small (128^2) so per-step FFTs are cheap.
"""

import numpy as np


def _wavenumbers(N, L=1.0):
    return 2 * np.pi * np.fft.fftfreq(N, d=L / N)  # includes the 2π scale


def vorticity_to_velocity(w, L=1.0):
    """Reconstruct divergence-free velocity from vorticity: -Δψ=w, u=(∂yψ, -∂xψ)."""
    N = w.shape[-1]
    kx = _wavenumbers(N, L)
    ky = _wavenumbers(N, L)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    wh = np.fft.fft2(w)
    psih = wh / K2
    u = np.real(np.fft.ifft2(1j * KY * psih))
    v = np.real(np.fft.ifft2(-1j * KX * psih))
    return u, v


def spectral_grad(f, L=1.0):
    """Spectral derivative of a periodic field; returns (∂x f, ∂y f)."""
    N = f.shape[-1]
    kx = _wavenumbers(N, L)
    ky = _wavenumbers(N, L)
    KX, KY = np.meshgrid(kx, ky)
    fh = np.fft.fft2(f)
    fx = np.real(np.fft.ifft2(1j * KX * fh))
    fy = np.real(np.fft.ifft2(1j * KY * fh))
    return fx, fy


# ---------- spectral ----------

def energy_spectrum(u, v, L=1.0):
    """Shell-averaged kinetic energy spectrum E(k). Returns (k_axis, E_k)."""
    N = u.shape[-1]
    kx = _wavenumbers(N, L)
    ky = _wavenumbers(N, L)
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX**2 + KY**2)
    uh = np.fft.fft2(u) / (N * N)
    vh = np.fft.fft2(v) / (N * N)
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    dk = 2 * np.pi / L
    kbin = np.round(kmag / dk).astype(int)
    E_k = np.bincount(kbin.ravel(), weights=e.ravel(), minlength=kbin.max() + 1)
    k_axis = np.arange(len(E_k)) * dk
    return k_axis, E_k


def spectral_band_error(u_p, v_p, u_t, v_t, L=1.0, n_bands=3):
    """Per-band relative spectrum error. The high band is the key spectral-bias signal."""
    _, Ep = energy_spectrum(u_p, v_p, L)
    _, Et = energy_spectrum(u_t, v_t, L)
    n = max(len(Ep), len(Et))
    Ep = np.pad(Ep, (0, n - len(Ep)))
    Et = np.pad(Et, (0, n - len(Et)))
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    names = ["low", "mid", "high"] if n_bands == 3 else [f"band{i}" for i in range(n_bands)]
    out = {}
    for i, name in enumerate(names):
        s = slice(edges[i], edges[i + 1])
        out[name] = np.linalg.norm(Ep[s] - Et[s]) / (np.linalg.norm(Et[s]) + 1e-12)
    return out


def spectral_centroid(u, v, L=1.0):
    """Spectral centroid k_c = Σ k E(k) / Σ E(k). Over-smoothing pulls it low."""
    k, E = energy_spectrum(u, v, L)
    return np.sum(k * E) / (np.sum(E) + 1e-12)


def enstrophy(w, L=1.0):
    """Enstrophy Z = 0.5 ∫ w² dA."""
    dA = (L / w.shape[-1]) ** 2
    return 0.5 * np.sum(w**2) * dA


# ---------- gradient ----------

def gradient_metrics(w_p, w_t, L=1.0):
    """Gradient-resolution metrics on the vorticity field. Ratios <1 mean over-smoothing."""
    gpx, gpy = spectral_grad(w_p, L)
    gp = np.sqrt(gpx**2 + gpy**2)
    gtx, gty = spectral_grad(w_t, L)
    gt = np.sqrt(gtx**2 + gty**2)
    grad_rmse = np.sqrt(np.mean((gpx - gtx) ** 2 + (gpy - gty) ** 2))
    grad_rel = grad_rmse / (np.sqrt(np.mean(gtx**2 + gty**2)) + 1e-12)
    dA = (L / w_p.shape[-1]) ** 2
    return {
        "grad_rel_err": grad_rel,
        "grad_max_ratio": gp.max() / (gt.max() + 1e-12),
        "grad_p99_ratio": np.percentile(gp, 99) / (np.percentile(gt, 99) + 1e-12),
        "TV_ratio": np.sum(gp) * dA / (np.sum(gt) * dA + 1e-12),
    }


# ---------- conservation ----------

def conservation_metrics(w_p, w_t, up, vp, ut, vt, L=1.0):
    """Discrete conservation errors vs ground truth (relative)."""
    dA = (L / w_p.shape[-1]) ** 2

    def integral(f):
        return f.sum() * dA

    def rel(a, b):
        return abs(a - b) / (abs(b) + 1e-12)

    # divergence of reconstructed velocity (incompressibility check)
    upx, _ = spectral_grad(up, L)
    _, vpy = spectral_grad(vp, L)
    div_rms = float(np.sqrt(np.mean((upx + vpy) ** 2)))

    Ep = 0.5 * np.sum(up**2 + vp**2) * dA
    Et = 0.5 * np.sum(ut**2 + vt**2) * dA
    Zp = enstrophy(w_p, L)
    Zt = enstrophy(w_t, L)
    Px_p, Py_p = integral(up), integral(vp)
    Px_t, Py_t = integral(ut), integral(vt)
    mom_err = np.hypot(Px_p - Px_t, Py_p - Py_t) / (np.hypot(Px_t, Py_t) + 1e-12)

    return {
        # Total vorticity ∫w is ~0 on typical NS ICs, so a relative error blows up; report
        # the absolute drift instead (markdown §2a). Kept as ``mass_abs_err``.
        "mass_abs_err": abs(integral(w_p) - integral(w_t)),
        "momentum_err": float(mom_err),
        "div_rms": div_rms,
        "energy_err": rel(Ep, Et),
        "enstrophy_err": rel(Zp, Zt),
    }


def ns_physical_report(w_pred, w_true, L=1.0):
    """All NS physical/spectral/gradient metrics for one (H, W) vorticity snapshot."""
    w_pred = np.asarray(w_pred, dtype=np.float64)
    w_true = np.asarray(w_true, dtype=np.float64)
    up, vp = vorticity_to_velocity(w_pred, L)
    ut, vt = vorticity_to_velocity(w_true, L)

    rep = {}
    rep.update(gradient_metrics(w_pred, w_true, L))
    rep.update({f"specerr_{k}": v for k, v in spectral_band_error(up, vp, ut, vt, L).items()})
    rep["spectral_centroid_ratio"] = spectral_centroid(up, vp, L) / (
        spectral_centroid(ut, vt, L) + 1e-12
    )
    rep.update(conservation_metrics(w_pred, w_true, up, vp, ut, vt, L))
    return rep


def scalar_field_report(f_pred, f_true, L=1.0):
    """Gradient / spectral metrics for a general 2D scalar field (e.g. shallow-water height).

    Same gradient and band-spectrum signals as :func:`ns_physical_report`, but without
    vorticity→velocity reconstruction. Spectrum uses the scalar power spectrum
    (``energy_spectrum(f, 0)``). Also reports relative mass (∫f) and L2-energy errors.
    FFT derivatives assume a periodic square of side ``L``; for open-boundary datasets
    (shallow water) treat comparisons as relative across models, not absolute physics.
    """
    f_pred = np.asarray(f_pred, dtype=np.float64)
    f_true = np.asarray(f_true, dtype=np.float64)
    zp = np.zeros_like(f_pred)
    zt = np.zeros_like(f_true)

    rep = {}
    rep.update(gradient_metrics(f_pred, f_true, L))
    rep.update({f"specerr_{k}": v for k, v in spectral_band_error(f_pred, zp, f_true, zt, L).items()})
    rep["spectral_centroid_ratio"] = spectral_centroid(f_pred, zp, L) / (
        spectral_centroid(f_true, zt, L) + 1e-12
    )
    dA = (L / f_pred.shape[-1]) ** 2
    mass_p = float(f_pred.sum() * dA)
    mass_t = float(f_true.sum() * dA)
    Ep = 0.5 * float(np.sum(f_pred**2) * dA)
    Et = 0.5 * float(np.sum(f_true**2) * dA)
    rep["mass_rel_err"] = abs(mass_p - mass_t) / (abs(mass_t) + 1e-12)
    rep["energy_err"] = abs(Ep - Et) / (abs(Et) + 1e-12)
    return rep


# =============================================================================
# 3-D versions — for volumetric fields (e.g. MHD 64³)
# =============================================================================

def spectral_grad_3d(f, L=1.0):
    """Spectral derivatives of a 3-D periodic field; returns (∂x f, ∂y f, ∂z f)."""
    D, H, W = f.shape
    kx = _wavenumbers(D, L)
    ky = _wavenumbers(H, L)
    kz = _wavenumbers(W, L)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    fh = np.fft.fftn(f)
    fx = np.real(np.fft.ifftn(1j * KX * fh))
    fy = np.real(np.fft.ifftn(1j * KY * fh))
    fz = np.real(np.fft.ifftn(1j * KZ * fh))
    return fx, fy, fz


def gradient_metrics_3d(f_p, f_t, L=1.0):
    """Gradient-resolution metrics for a 3-D field. Ratios <1 mean over-smoothing."""
    gpx, gpy, gpz = spectral_grad_3d(f_p, L)
    gtx, gty, gtz = spectral_grad_3d(f_t, L)
    gp = np.sqrt(gpx**2 + gpy**2 + gpz**2)
    gt = np.sqrt(gtx**2 + gty**2 + gtz**2)
    grad_rmse = np.sqrt(np.mean((gpx - gtx)**2 + (gpy - gty)**2 + (gpz - gtz)**2))
    grad_rel = grad_rmse / (np.sqrt(np.mean(gtx**2 + gty**2 + gtz**2)) + 1e-12)
    dV = (L / f_p.shape[-1]) ** 3
    return {
        "grad_rel_err": grad_rel,
        "grad_max_ratio": gp.max() / (gt.max() + 1e-12),
        "grad_p99_ratio": np.percentile(gp, 99) / (np.percentile(gt, 99) + 1e-12),
        "TV_ratio": np.sum(gp) * dV / (np.sum(gt) * dV + 1e-12),
    }


def energy_spectrum_3d(f, L=1.0):
    """Shell-averaged scalar energy spectrum E(k) for a 3-D periodic field."""
    N = f.shape[-1]
    k1d = _wavenumbers(N, L)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    fh = np.fft.fftn(f) / (N**3)
    e = 0.5 * np.abs(fh) ** 2
    dk = 2 * np.pi / L
    kbin = np.round(kmag / dk).astype(int)
    E_k = np.bincount(kbin.ravel(), weights=e.ravel(), minlength=kbin.max() + 1)
    k_axis = np.arange(len(E_k)) * dk
    return k_axis, E_k


def spectral_band_error_3d(f_p, f_t, L=1.0, n_bands=3):
    """Per-band relative spectrum error for a 3-D scalar field."""
    _, Ep = energy_spectrum_3d(f_p, L)
    _, Et = energy_spectrum_3d(f_t, L)
    n = max(len(Ep), len(Et))
    Ep = np.pad(Ep, (0, n - len(Ep)))
    Et = np.pad(Et, (0, n - len(Et)))
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    names = ["low", "mid", "high"] if n_bands == 3 else [f"band{i}" for i in range(n_bands)]
    out = {}
    for i, name in enumerate(names):
        s = slice(edges[i], edges[i + 1])
        out[name] = np.linalg.norm(Ep[s] - Et[s]) / (np.linalg.norm(Et[s]) + 1e-12)
    return out


def spectral_centroid_3d(f, L=1.0):
    """Spectral centroid k_c = Σ k E(k) / Σ E(k) for a 3-D field."""
    k, E = energy_spectrum_3d(f, L)
    return np.sum(k * E) / (np.sum(E) + 1e-12)


def scalar_field_report_3d(f_pred, f_true, L=1.0):
    """Gradient / spectral metrics for a general 3-D scalar field (e.g. one MHD component).

    Consistent with :func:`scalar_field_report` but uses 3-D FFT and volumetric gradients.
    Averages are over the caller's (batch × time × channel) loop; each call receives a
    single (D, H, W) snapshot of one field component.
    """
    f_pred = np.asarray(f_pred, dtype=np.float64)
    f_true = np.asarray(f_true, dtype=np.float64)
    rep = {}
    rep.update(gradient_metrics_3d(f_pred, f_true, L))
    rep.update({f"specerr_{k}": v for k, v in spectral_band_error_3d(f_pred, f_true, L).items()})
    rep["spectral_centroid_ratio"] = spectral_centroid_3d(f_pred, L) / (
        spectral_centroid_3d(f_true, L) + 1e-12
    )
    dV = (L / f_pred.shape[-1]) ** 3
    mass_p = float(f_pred.sum() * dV)
    mass_t = float(f_true.sum() * dV)
    Ep = 0.5 * float(np.sum(f_pred**2) * dV)
    Et = 0.5 * float(np.sum(f_true**2) * dV)
    rep["mass_rel_err"] = abs(mass_p - mass_t) / (abs(mass_t) + 1e-12)
    rep["energy_err"] = abs(Ep - Et) / (abs(Et) + 1e-12)
    return rep
