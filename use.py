import numpy as np

# ============================================================
# ===  GRAD / DIV (adjoints exacts, Neumann)
# ============================================================
def grad_forward_neumann(x, gx, gy):
    """
    Gradient avant avec conditions de Neumann (bord = 0).
    gx, gy sont écrits in-place.
    """
    gx[:-1, :] = x[1:, :] - x[:-1, :]
    gx[-1, :] = 0.0

    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:, -1] = 0.0


def div_backward_neumann(px, py, div):
    """
    Divergence adjoint exact de grad_forward_neumann.
    div est écrit in-place.
    """
    div.fill(0.0)

    div[1:, :] += px[1:, :] - px[:-1, :]
    div[0, :] += px[0, :]

    div[:, 1:] += py[:, 1:] - py[:, :-1]
    div[:, 0] += py[:, 0]


# ============================================================
# ===  PROX POISSON (branchless, stable)
# ============================================================
def prox_poisson(eta, tau, y):
    """
    Prox de la fidélité Poisson : argmin_x tau * (x - y*log x) + 0.5 * (x - eta)^2
    (avec convention y*log x = 0 si y=0 ou x=0).
    Formule fermée, vectorisée, branchless.
    """
    y = np.maximum(y, 0.0)

    # b = eta - tau
    b = eta - tau
    disc = b * b + 4.0 * tau * y
    disc = np.maximum(disc, 0.0)

    # Formule fermée
    x = 0.5 * (b + np.sqrt(disc))

    # Clamp pour éviter les résidus négatifs numériques
    return np.maximum(x, 0.0)


# ============================================================
# ===  PROJECTION TV ISOTROPE
# ============================================================
def project_iso(px, py, ltv):
    """
    Projection isotrope sur { (px,py) : sqrt(px^2 + py^2) <= ltv }.
    In-place, stable.
    """
    if ltv <= 0:
        px.fill(0.0)
        py.fill(0.0)
        return

    nrm = np.sqrt(px * px + py * py)
    # scale = max(1, nrm / ltv) => on divise seulement si nrm > ltv
    # évite les divisions inutiles et les problèmes de nrm=0
    mask = nrm > ltv
    if not np.any(mask):
        return

    scale = (nrm[mask] / ltv)
    px[mask] /= scale
    py[mask] /= scale


# ============================================================
# ===  ESTIMATION DE ||K|| PAR POWER ITERATION
# ============================================================
def estimate_K_norm(shape, iters=20, seed=0):
    """
    Estime ||K|| où K = grad (Neumann) via power iteration sur K*K*.
    Plus stable que la version naïve, déterministe via seed.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    div = np.zeros_like(x)

    # Normalisation initiale
    nrm = np.linalg.norm(x)
    if nrm == 0:
        x[...] = 1.0
    else:
        x /= nrm

    for _ in range(iters):
        # u = K x
        grad_forward_neumann(x, gx, gy)
        # v = K* u = div(gx, gy)
        div_backward_neumann(gx, gy, div)

        nrm = np.linalg.norm(div)
        if nrm == 0:
            break
        x[...] = div / nrm

    # Norme finale : ||K|| ≈ ||K x||
    grad_forward_neumann(x, gx, gy)
    return np.sqrt(np.linalg.norm(gx) ** 2 + np.linalg.norm(gy) ** 2)


# ============================================================
# ===  PDHG POISSON + TV ISOTROPE (version 10/10)
# ============================================================
def pdhg_poisson_tv(
    y,
    ltv=0.1,
    tau=None,
    sigma=None,
    theta=1.0,
    iters=300,
    tol=0.0,
    verbose=False,
    return_history=False,
    seed_norm=0,
):
    """
    Résout : min_x  sum_i ( x_i - y_i log x_i ) + ltv * TV_iso(x)
    avec TV isotrope et conditions de Neumann.

    Paramètres :
        y : image d'observation (Poisson)
        ltv : poids de la TV
        tau, sigma : pas primal/dual (si None, choisis automatiquement)
        theta : paramètre d'extrapolation (1.0 = standard PDHG)
        iters : nombre max d'itérations
        tol : critère d'arrêt relatif sur ||x^{k+1} - x^k||
        verbose : affiche un log toutes les 50 itérations
        return_history : si True, renvoie aussi l'historique des résidus
        seed_norm : seed pour l'estimation de ||K||

    Retour :
        x (et éventuellement history dict si return_history=True)
    """
    # --- Prétraitement ---
    y = np.maximum(y.astype(float), 0.0)
    x = y.copy()
    xbar = x.copy()

    px = np.zeros_like(x)
    py = np.zeros_like(x)

    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    div = np.zeros_like(x)

    x_old = np.zeros_like(x)

    # --- Norme opérateur K ---
    K_norm = estimate_K_norm(x.shape, seed=seed_norm)

    # --- Pas automatiques (symétriques) ---
    if tau is None or sigma is None:
        # 0.99 / ||K|| est un choix standard stable
        tau = 0.99 / K_norm
        sigma = 0.99 / K_norm

    history = {
        "primal_res": [],
    } if return_history else None

    # --- Boucle PDHG ---
    for k in range(iters):

        # --- Dual update ---
        grad_forward_neumann(xbar, gx, gy)
        px += sigma * gx
        py += sigma * gy
        project_iso(px, py, ltv)

        # --- Primal update ---
        div_backward_neumann(px, py, div)
        x_old[...] = x  # évite une allocation

        eta = x - tau * div
        x = prox_poisson(eta, tau, y)

        # --- Extrapolation ---
        xbar[...] = x + theta * (x - x_old)
        # Clamp pour rester dans le domaine (x >= 0)
        np.maximum(xbar, 0.0, out=xbar)

        # --- Critère d'arrêt (primal) ---
        if tol > 0:
            diff = x - x_old
            primal = np.linalg.norm(diff)
            denom = np.linalg.norm(x_old) + 1e-12
            rel = primal / denom

            if return_history:
                history["primal_res"].append(rel)

            if verbose and (k % 50 == 0 or k == iters - 1):
                print(f"Iter {k:4d}: rel_primal = {rel:.3e}")

            if rel < tol:
                if verbose:
                    print(f"Arrêt à l'itération {k} (tol atteinte : {rel:.3e})")
                break
        else:
            if verbose and k % 50 == 0:
                # on calcule quand même pour le log
                diff = x - x_old
                primal = np.linalg.norm(diff)
                denom = np.linalg.norm(x_old) + 1e-12
                rel = primal / denom
                print(f"Iter {k:4d}: rel_primal = {rel:.3e}")

    if return_history:
        return x, history
    return x
