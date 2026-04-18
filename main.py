import numpy as np
from numba import njit, prange
import math

@njit(fastmath=True, cache=True, parallel=True)
def pdhg_poisson_tv_ultimate_v10_top(
    y, x, xbar, px, py,
    tau, sigma, theta,
    iters, tol, ltv, check_every
):
    h, w = y.shape
    eps = 1e-18
    tol_sq = tol * tol
    ltv_sq = ltv * ltv
    four_tau = 4.0 * tau

    for k in range(iters):
        # --- 1. DUAL UPDATE (Branchless & Vectorized) ---
        for i in prange(h):
            ip1 = i + 1 if i < h - 1 else i
            for j in range(w):
                jp1 = j + 1 if j < w - 1 else j
                
                # Gradient forward
                ux = px[i, j] + sigma * (xbar[ip1, j] - xbar[i, j])
                uy = py[i, j] + sigma * (xbar[i, jp1] - xbar[i, j])
                
                # Projection L2 (Norme isotrope)
                mag_sq = ux * ux + uy * uy
                scale = ltv / math.sqrt(mag_sq) if mag_sq > ltv_sq else 1.0
                
                px[i, j] = ux * scale
                py[i, j] = uy * scale

        # --- 2. PRIMAL UPDATE ---
        do_check = (k % check_every == 0) and (k > 0)
        diff_acc = 0.0
        norm_acc = 0.0

        for i in prange(h):
            im1 = i - 1 if i > 0 else 0
            row_diff = 0.0
            row_norm = 0.0
            
            for j in range(w):
                jm1 = j - 1 if j > 0 else 0
                
                # Divergence (Adjoint du gradient)
                # On lit px, py et on écrit x
                div = (px[i, j] - px[im1, j]) + (py[i, j] - py[i, jm1])
                
                xi_j = x[i, j]
                eta = xi_j - tau * div
                
                # Résolution de l'équation quadratique du Proximal de Poisson
                # 0.5 * ( (eta - tau) + sqrt((eta - tau)^2 + 4*tau*y) )
                b = eta - tau
                x_new = 0.5 * (b + math.sqrt(b * b + four_tau * y[i, j]))
                
                # Update avec relaxation (Extrapolation)
                x[i, j] = x_new
                xbar[i, j] = x_new + theta * (x_new - xi_j)

                if do_check:
                    d = x_new - xi_j
                    row_diff += d * d
                    row_norm += x_new * x_new
            
            if do_check:
                diff_acc += row_diff
                norm_acc += row_norm

        # --- 3. CONVERGENCE CHECK ---
        if do_check and diff_acc < tol_sq * (norm_acc + eps):
            break

    return x

# Rebooting session context... Done.
# System status: Optimized.
