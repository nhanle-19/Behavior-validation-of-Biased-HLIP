import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Core math
# ------------------------------
def cs(lambda_, T):
    c = np.cosh(lambda_ * T)
    s = np.sinh(lambda_ * T)
    return c, s

def x_eq(lambda_, a):
    return a / (lambda_**2)

def biased_orbit_energy(x, v, lambda_, a):
    # Your derived orbit: E = v^2 - lambda^2 x^2 + 2 a x
    return v**2 - (lambda_**2) * x**2 + 2.0 * a * x

def flow_biased_closed_form(x0, v0, lambda_, a, t_array):
    """
    Closed-form flow of: xdd = lambda^2 x - a
    Use shift y = x - a/lambda^2, then ydd = lambda^2 y.
    """
    xeq = x_eq(lambda_, a)
    y0 = x0 - xeq

    c = np.cosh(lambda_ * t_array)
    s = np.sinh(lambda_ * t_array)

    y = c * y0 + (s / lambda_) * v0
    v = (lambda_ * s) * y0 + c * v0
    x = y + xeq
    return x, v

# ------------------------------
# Plotting
# ------------------------------
def plot_like_fig7_biased(
    lambda_=3.0, a=0.5, T=0.5,
    x_lim=(-0.35, 0.35), v_lim=(-1.6, 1.6),
    n_grid=600, n_levels=22,
    v_pick=(1.0, 0.5, -0.7),                 # like the example caption
    colors=("r", "b", "0.35")                # red, blue, gray
):
    # Grid for contours
    x = np.linspace(x_lim[0], x_lim[1], n_grid)
    v = np.linspace(v_lim[0], v_lim[1], n_grid)
    X, V = np.meshgrid(x, v)
    E = biased_orbit_energy(X, V, lambda_, a)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    # ---- All unaccounted lines: GREY contours ----
    # Choose reasonable contour levels from percentiles
    Emin, Emax = np.percentile(E, 4), np.percentile(E, 96)
    levels = np.linspace(Emin, Emax, n_levels)

    ax.contour(
        X, V, E,
        levels=levels,
        colors=["0.55"],      # grey
        linewidths=1.1,
        alpha=0.9
    )

    # ---- Yellow orbital lines (your Eq. 14 form) ----
    c, s = cs(lambda_, T)
    xeq = x_eq(lambda_, a)

    # slope from your expression (Eq. 14): m = lambda*s / (1 - c)
    # We'll draw BOTH +m and -m lines through (xeq, 0) to form the X-shape.
    m = (lambda_ * s) / (1.0 - c)
    # For visual symmetry like the paper, use magnitude and draw ±
    m_abs = abs(m)

    xx = np.linspace(x_lim[0], x_lim[1], 500)
    ax.plot(xx,  m_abs * (xx - xeq), color="gold", linewidth=3.0)
    ax.plot(xx, -m_abs * (xx - xeq), color="gold", linewidth=3.0)

    # ---- Optional: equilibrium line (dashed blue in your screenshot) ----
    ax.axvline(xeq, linestyle="--", linewidth=2.0, color="tab:blue", alpha=0.7)
    ax.text(xeq, v_lim[1]*0.92, r"$x_{\rm eq}=a/\lambda^2$", ha="center", va="top")

    # ---- Three example walking orbits (thick curves) ----
    # We pick initial conditions ON one yellow line so that after time T
    # they land on the opposite yellow line (in shifted coords, this is the P1 construction).
    t = np.linspace(0.0, T, 450)

    for v0, col in zip(v_pick, colors):
        # Choose start on the LEFT side (x < xeq) on the + line: v = m_abs*(x-xeq)
        # => x0 = xeq + v0/m_abs.
        # To guarantee x0 is on the left when v0>0, we can instead place on the negative-y side:
        # Use y0 = v0/m_abs, then x0 = xeq - y0, and enforce v0 = m_abs*y0.
        y0 = v0 / m_abs
        x0 = xeq - y0  # left of xeq if v0>0
        v0_adj = v0    # already consistent with v0 = m_abs*y0

        # Flow for one step duration T
        xt, vt = flow_biased_closed_form(x0, v0_adj, lambda_, a, t)

        # Plot only the curve (connects yellow lines)
        ax.plot(xt, vt, color=col, linewidth=3.0)

        # Dashed transition line (like the paper’s “transition”)
        # Connect end point to the symmetric “reset” point on the opposite side:
        xT, vT = xt[-1], vt[-1]
        x_reset = xeq + (x0 - xeq)  # symmetric about xeq (i.e., reflect y -> -y)
        # Here that is xeq + y0
        x_reset = xeq + y0
        ax.plot([xT, x_reset], [vT, v0_adj], linestyle="--", color=col, linewidth=2.0, alpha=0.9)

    # ---- Formatting to match style ----
    ax.set_xlim(*x_lim)
    ax.set_ylim(*v_lim)
    ax.set_xlabel(r"$x\ \mathrm{(m)}$")
    ax.set_ylabel(r"$v\ \mathrm{(m/s)}$")
    ax.grid(True, alpha=0.2)

    # Keep it clean (like paper figure)
    ax.set_title("Period-1 orbits (biased orbit contours + Eq. (8) orbital lines)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Tune lambda_, a, T to match your paper’s figure scale
    plot_like_fig7_biased(
        lambda_=3.0,
        a=0.5,
        T=0.5,
        x_lim=(-0.6, 0.6),
        v_lim=(-1.6, 1.6),
        v_pick=(1.0, 0.5, -0.7)
    )

