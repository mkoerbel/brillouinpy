# Installation

The instructions below use [Anaconda](https://www.anaconda.com/download/success) as
the Python distribution, recommended if you are new to Python. Any other Python
≥ 3.11 environment works just as well.

```{note}
Unless you have a specific reason to upgrade, keep your conda installation as-is -
upgrading conda itself can break existing environments.
```

## Prerequisites

1. Install Anaconda: [download](https://www.anaconda.com/download/success).
2. Create a new conda environment: `conda create --name <env-name>`.
3. Activate it: `conda activate <env-name>`.
4. Install pip and git into it: `conda install pip git`.

Then install BrillouinPy one of two ways.

## Install from the Git repository

Recommended if you only use the package and want to stay on the latest version.

```bash
pip install git+https://github.com/timm-landes/brillouinpy.git
```

## Install from a local clone

Recommended if you intend to modify the package code itself.

```bash
git clone https://github.com/timm-landes/brillouinpy.git
cd brillouinpy
pip install .
# or, for an editable install that picks up local changes without reinstalling:
pip install -e .
```

```{admonition} LUH-internal
:class: note

From inside the university network you can use the GitLab mirror instead by
replacing the URL with
`git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinpy.git` (requires a
GitLab SSH key registered with your LUH account). It is not reachable outside the
LUH network/SSO.
```

## Optional extras

Some interoperability formats need extra packages, installed with the usual
`[extra]` syntax:

```bash
pip install "brillouinpy[brim] @ git+https://github.com/timm-landes/brillouinpy.git"
```

- `brim` - read/write the [brim](https://github.com/brillouin-imaging/Brillouin-standard-file)
  format (`brimfile`).
- `hdf5_bls` - read/write the [HDF5_BLS](https://github.com/bio-brillouin/HDF5_BLS)
  format.

## Updating

```bash
conda activate <env-name>
pip install --upgrade git+https://github.com/timm-landes/brillouinpy.git   # if installed from git
pip install --upgrade .                                                    # if installed from a local clone
```

## Using the environment in Spyder

Install the Spyder kernels into the environment (via conda, not pip):

```bash
conda install spyder-kernels
# Spyder 5.x needs a pinned version instead:
conda install spyder-kernels=2.5
```

Then, in Spyder, open a new console on that environment (Spyder 6), or set it as
the default environment via the interpreter selector in the bottom status bar
(Spyder 5).
