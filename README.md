# Biased HLIP Preliminary Demo

This repository contains a preliminary numerical demonstration of biased H-LIP
(Hybrid Linear Inverted Pendulum) ideas for simple five-link walking
simulations. The current focus is a compact, reproducible workspace for
experiments with slope/acceleration bias terms and a simple Pinocchio URDF
model.

The code is research/prototype quality. It is intended to show the main
simulation ideas and produce qualitative plots/animations, not to provide a
finished controller package.

## Repository Layout

- `examples/full_order_biased_hlip_acceleration.py` - main five-link full-order
  biased HLIP walking demonstration using the URDF model.
- `examples/hlip_point_mass_check.py` - reduced-order biased H-LIP consistency
  check using the same pre-impact stepping convention as the full-order demo.
- `examples/biased_orbit_plot.py` - helper script for biased orbit contour
  visualization.
- `models/five_link_walker.urdf` - simple five-link walker model.

Generated plots, screenshots, documents, caches, and local virtual environments
are intentionally excluded from version control.

## Setup

Use a Python environment with NumPy, Matplotlib, and Pinocchio available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Depending on your platform, Pinocchio may be easiest to install through conda:

```bash
conda install -c conda-forge pinocchio
pip install numpy matplotlib
```

## Run

From the repository root:

```bash
python examples/full_order_biased_hlip_acceleration.py
```

Optional helper demos:

```bash
python examples/hlip_point_mass_check.py
python examples/biased_orbit_plot.py
```

The scripts open Matplotlib windows for plots/animations. For headless syntax
checks, use:

```bash
MPLBACKEND=Agg python -m py_compile examples/*.py
```

## Notes

- The main demonstration currently uses the acceleration-bias setup in
  `examples/full_order_biased_hlip_acceleration.py`.
- The reduced-order check is a consistency tool for the biased fixed point and
  foot-placement convention; the five-link script remains the main result.
- The URDF path is resolved relative to the repository root, so the main script
  does not depend on the shell's current working directory.
- This is a preliminary biased HLIP demonstration. Parameters, gains, and
  validation criteria should be revisited before treating the results as final.
