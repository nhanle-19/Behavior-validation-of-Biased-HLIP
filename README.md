# Biased H-LIP Methodology Showcase

This repository is a small demonstration of the biased H-LIP methodology for
walking under horizontal bias effects, such as slope or accelerating ground. It
is meant to show the idea and simulation structure, not to represent current
active work, a finished controller package, or a complete validation study.

The examples illustrate how a biased H-LIP equilibrium and pre-impact
foot-placement law can be connected to a simple five-link walking model. The
scripts are intentionally lightweight so the methodology can be inspected,
modified, and reproduced.

## Contents

- `examples/full_order_biased_hlip_acceleration.py` - five-link Pinocchio
  demonstration using biased H-LIP foot placement.
- `examples/hlip_point_mass_check.py` - reduced-order biased H-LIP consistency
  check using the same pre-impact stepping convention.
- `examples/biased_orbit_plot.py` - biased orbit contour visualization.
- `models/five_link_walker.urdf` - simple five-link walker model.

Generated plots, screenshots, documents, caches, and local virtual environments
are intentionally excluded from version control.

## Setup

Use a Python environment with NumPy, Matplotlib, and Pinocchio available.

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

Reduced-order consistency check and orbit visualization:

```bash
python examples/hlip_point_mass_check.py
python examples/biased_orbit_plot.py
```

The scripts open Matplotlib windows for plots/animations. For headless syntax
checks, use:

```bash
MPLBACKEND=Agg python -m py_compile examples/*.py
```

## Methodology Notes

- The five-link example uses the acceleration-bias setup in
  `examples/full_order_biased_hlip_acceleration.py`.
- The reduced-order check is included to show the biased fixed point and
  pre-impact foot-placement convention in isolation.
- The URDF path is resolved relative to the repository root, so the main script
  does not depend on the shell's current working directory.
- Parameters and gains are example values chosen for demonstration. They should
  not be interpreted as final tuned results.
