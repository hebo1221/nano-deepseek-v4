# Installation

Python 3.10 or newer and PyTorch 2.4 or newer are required. On Linux, install
PyTorch before this package. That keeps a CPU environment on the CPU wheel
index instead of allowing the default PyPI resolver to select PyTorch's much
larger CUDA dependency set.

## CPU — published package

Create an isolated environment, install the CPU PyTorch wheel, and then install
the package from PyPI:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.4"
python -m pip install nano-deepseek-v4
nano-deepseek-v4-demo
```

The two install commands are intentional. `--extra-index-url` is not used:
mixing CPU and accelerator indexes would leave the selected PyTorch build to
resolver ordering.

## CPU — current source

Use the same PyTorch-first order for an editable checkout:

```bash
git clone https://github.com/hebo1221/nano-deepseek-v4.git
cd nano-deepseek-v4
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.4"
python -m pip install -e .
```

Check the installation without downloading a checkpoint:

```bash
nano-deepseek-v4 --version
nano-deepseek-v4 demo
nano-deepseek-v4 tour
nano-deepseek-v4 dspark
nano-deepseek-v4 dspark-scheduler
# Equivalent module entry point:
python -m nano_deepseek_v4 demo
```

Both DSpark commands are CPU-only, no-download checks. `dspark` covers the tiny
draft equations; `dspark-scheduler` covers Algorithm 1 and Section 5.2
arithmetic on packaged calibrated probabilities and synthetic SPS tables.
`tour` traces real tensor shapes through every attention and routing family in
the tiny model, then points to the
[modeling walkthrough](modeling-walkthrough.md). It is also CPU-only and
no-download.

The current source keeps the legacy commands such as
`nano-deepseek-v4-demo` for compatibility.

## CUDA or another accelerator

Install the PyTorch build selected for the operating system, accelerator, and
driver by the
[official PyTorch installer](https://pytorch.org/get-started/locally/), verify
that build, and only then install this package:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install nano-deepseek-v4
```

For a source checkout, replace the final command with
`python -m pip install -e .`. This project does not replace an existing
PyTorch build or choose a CUDA runtime for the user.

## Optional capabilities

Extras add only the capability named here; they do not change the PyTorch build
selected above.

| Extra | Availability | Adds |
| --- | --- | --- |
| `official` | PyPI 0.2.0 and current source | Hugging Face Hub access for revision-pinned remote inspection, generation, and Flash-0731 receipt replay |
| `parity` | Current source | The pinned Transformers version used by the differential verifier |
| `dev` | PyPI 0.2.0 and current source | Tests, coverage, lint, typing, and package-build tools |
| `notebook` | PyPI 0.2.0 and current source | Jupyter runtime for notebooks opened from the repository or source archive; notebook files are not installed into `site-packages` |

For example, after installing PyTorch:

```bash
# Published package (0.2.0 has the official extra, but not parity)
python -m pip install "nano-deepseek-v4[official]==0.2.0"

# Current source checkout
python -m pip install -e ".[dev,official,parity]"
```

To run the CPU-first tutorial from a source checkout, install its runtime and
open the checked-in file:

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/01_flash_inference.ipynb
```

The default path is local and does not download a checkpoint. The optional
official-checkpoint appendix requires the `official` extra and makes its
network and storage cost explicit. A wheel installation provides the
`notebook` runtime dependencies but does not copy repository notebooks into
the environment. Do not pair an older wheel with the mutable `main` notebook;
download the notebook from the tag matching the installed package:

```bash
NANO_VERSION="$(python -c 'from importlib.metadata import version; print(version("nano-deepseek-v4"))')"
python -m pip install "nano-deepseek-v4[notebook]==${NANO_VERSION}"
curl --fail --location --output 01_flash_inference.ipynb \
  "https://raw.githubusercontent.com/hebo1221/nano-deepseek-v4/v${NANO_VERSION}/notebooks/01_flash_inference.ipynb"
jupyter lab 01_flash_inference.ipynb
```

The matching source archive is an equivalent source for the notebook.

Run `python -m pip check` after changing environments. The package version alone
cannot distinguish a published install from an Unreleased editable checkout;
`python -m pip show nano-deepseek-v4` reports an `Editable project location`
for the latter.
