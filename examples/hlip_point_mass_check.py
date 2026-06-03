"""
Reduced-order biased H-LIP consistency check.

This script uses the same state convention as the full-order demo:
  - x = [p, v] is the CoM position/velocity relative to the current stance foot.
  - SSP dynamics are p_ddot = lambda^2 p - g_eff_x.
  - The stepping law is applied to the pre-impact state.
  - After touchdown, the new stance becomes the origin of the next step.

It is not a replacement for the five-link simulation; it is a small model used
to check the biased fixed point and foot-placement convention.
"""

import numpy as np
import matplotlib.pyplot as plt


def coth(x):
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.where(ax < 1e-10, np.where(x >= 0, 1e10, -1e10), np.cosh(x) / np.sinh(x))
    return float(out) if np.isscalar(x) else out


def biased_hlip_flow(x0, duration, lam, g_eff_x):
    """Closed-form flow for p_ddot = lambda^2 p - g_eff_x."""
    p0, v0 = np.asarray(x0, dtype=float).ravel()
    xeq = g_eff_x / (lam**2)
    y0 = p0 - xeq
    c = np.cosh(lam * duration)
    s = np.sinh(lam * duration)

    p = xeq + c * y0 + (s / lam) * v0
    v = lam * s * y0 + c * v0
    return np.array([p, v], dtype=float)


def biased_preimpact_fixed_point(Ts, Td, lam, vd, g_eff_x):
    """Pre-impact biased H-LIP fixed point used by the full-order demo."""
    T = Ts + Td
    u_star = vd * T
    xeq = g_eff_x / (lam**2)
    sigma = lam * coth(lam * T / 2.0)
    p_star_pre = xeq + u_star / (2.0 + Td * sigma)
    v_star_pre = u_star * sigma / (2.0 + Td * sigma)
    x_star_pre = np.array([p_star_pre, v_star_pre], dtype=float)
    return u_star, xeq, x_star_pre


def deadbeat_gain(Ts, Td, lam):
    """H-LIP pre-impact foot placement gain."""
    return np.array([[1.0, Td + (1.0 / lam) * coth(lam * Ts)]], dtype=float)


def simulate_biased_hlip(
    *,
    z0=0.73,
    g_eff_x=-0.9,
    g_eff_z=9.81,
    Ts=0.35,
    Td=0.0,
    vd=0.2,
    num_steps=20,
    initial_error=(0.03, -0.10),
    samples_per_step=30,
):
    lam = np.sqrt(g_eff_z / z0)
    u_star, xeq, x_star_pre = biased_preimpact_fixed_point(Ts, Td, lam, vd, g_eff_x)
    K = deadbeat_gain(Ts, Td, lam)

    # Post-impact fixed point in the new stance frame.
    p_post_star = x_star_pre[0] + Td * x_star_pre[1] - u_star
    x = np.array([p_post_star, x_star_pre[1]], dtype=float) + np.array(initial_error, dtype=float)

    stance_world = 0.0
    states_post = [x.copy()]
    states_pre = []
    stance_positions = [stance_world]
    step_lengths = []
    anim_com_x = []
    anim_foot_x = []
    anim_time = []

    for step_idx in range(num_steps):
        sample_times = np.linspace(0.0, Ts, samples_per_step, endpoint=False)
        for sample_time in sample_times:
            x_sample = biased_hlip_flow(x, sample_time, lam, g_eff_x)
            anim_com_x.append(stance_world + x_sample[0])
            anim_foot_x.append(stance_world)
            anim_time.append(step_idx * (Ts + Td) + sample_time)

        x_pre = biased_hlip_flow(x, Ts, lam, g_eff_x)
        uk = float(u_star + (K @ (x_pre - x_star_pre))[0])

        states_pre.append(x_pre.copy())
        step_lengths.append(uk)

        # Optional DSP drift is approximated as constant velocity, matching the
        # fixed-point convention used in the stepping law.
        x = np.array([x_pre[0] + Td * x_pre[1] - uk, x_pre[1]], dtype=float)
        stance_world += uk
        stance_positions.append(stance_world)
        states_post.append(x.copy())

    anim_com_x.append(stance_world + states_post[-1][0])
    anim_foot_x.append(stance_world)
    anim_time.append(num_steps * (Ts + Td))

    return {
        "lambda": lam,
        "u_star": u_star,
        "xeq": xeq,
        "x_star_pre": x_star_pre,
        "K": K,
        "states_post": np.asarray(states_post),
        "states_pre": np.asarray(states_pre),
        "stance_positions": np.asarray(stance_positions),
        "step_lengths": np.asarray(step_lengths),
        "anim_com_x": np.asarray(anim_com_x),
        "anim_foot_x": np.asarray(anim_foot_x),
        "anim_time": np.asarray(anim_time),
        "z0": z0,
    }


def plot_results(result):
    steps = np.arange(1, len(result["step_lengths"]) + 1)
    com_world_post = result["states_post"][:, 0] + result["stance_positions"]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    axs[0].plot(steps, result["step_lengths"], "o-", label=r"$u_k$")
    axs[0].axhline(result["u_star"], linestyle="--", color="k", label=r"$u^*$")
    axs[0].set_ylabel("step length (m)")
    axs[0].set_title("Biased H-LIP foot-placement convergence")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    sample = np.arange(len(com_world_post))
    axs[1].plot(sample, com_world_post, "o-", label="CoM post-impact")
    axs[1].plot(sample, result["stance_positions"], "s-", label="stance")
    axs[1].set_xlabel("step index")
    axs[1].set_ylabel("world x (m)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def animate_com_foot(result, pause=0.02):
    """Animate the reduced-order CoM and stance foot in the x-z plane."""
    x_pos = result["anim_com_x"]
    foot_pos = result["anim_foot_x"]
    z0 = result["z0"]

    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Reduced-order biased H-LIP: CoM and stance foot")

    all_x = np.concatenate([x_pos, foot_pos])
    pad = 0.1 * max(float(np.max(all_x) - np.min(all_x)), 1e-6)
    ax.set_xlim(float(np.min(all_x) - pad), float(np.max(all_x) + pad))
    ax.set_ylim(-0.15 * z0, 1.2 * z0)
    ax.plot([np.min(all_x) - pad, np.max(all_x) + pad], [0.0, 0.0], "k-", linewidth=1)
    ax.plot(x_pos, np.full_like(x_pos, z0), ":", alpha=0.5, label="CoM path")
    ax.plot(foot_pos, np.zeros_like(foot_pos), ":", alpha=0.5, label="stance path")

    (com_marker,) = ax.plot([x_pos[0]], [z0], "o", markersize=8, label="CoM")
    (foot_marker,) = ax.plot([foot_pos[0]], [0.0], "s", markersize=8, label="stance foot")
    (leg_line,) = ax.plot([foot_pos[0], x_pos[0]], [0.0, z0], "-", linewidth=2)
    ax.legend(loc="best")

    fig.canvas.draw()
    fig.canvas.flush_events()

    for com_x, foot_x in zip(x_pos, foot_pos):
        com_marker.set_data([com_x], [z0])
        foot_marker.set_data([foot_x], [0.0])
        leg_line.set_data([foot_x, com_x], [0.0, z0])
        fig.canvas.draw_idle()
        plt.pause(pause)

    plt.ioff()
    plt.show()


def main():
    result = simulate_biased_hlip()

    print(f"lambda = {result['lambda']:.4f}")
    print(f"x_eq = {result['xeq']:.4f} m")
    print(f"u* = {result['u_star']:.4f} m")
    print(f"pre-impact fixed point = {result['x_star_pre']}")
    print(f"K = {result['K']}")
    print("\nstep | p_pre (m) | v_pre (m/s) | u_k (m)")
    for i, (x_pre, uk) in enumerate(zip(result["states_pre"], result["step_lengths"]), start=1):
        print(f"{i:4d} | {x_pre[0]:9.4f} | {x_pre[1]:11.4f} | {uk:7.4f}")

    plot_results(result)
    animate_com_foot(result)


if __name__ == "__main__":
    main()
