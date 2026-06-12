# `numcompute_stream`

`numcompute_stream` is a scientific computing toolkit developed with plain python and [NumPy](https://github.com/numpy/numpy) library with the support of streaming data processing, online-learning and real-time model evaluation.

---

## Table of Contents

- [`numcompute_stream`](#numcompute_stream)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [API Overview](#api-overview)
  - [Examples](#examples)

---

## Installation

In addition to core Python and NumPy, `numcompute_stream` uses these other libraries for supporting functions:

- [flake8](https://github.com/PyCQA/flake8): code linting (PEP8)
- [pytest](https://github.com/pytest-dev/pytest), [pytest-cov](https://github.com/pytest-dev/pytest-cov): automated tests and coverage reporting
- [microbench](https://github.com/alubbock/microbench), [pandas](https://github.com/pandas-dev/pandas), [matplotlib](https://github.com/matplotlib/matplotlib): executing benchmarks and dispaying results

### Prerequisites

- Python 3.13+
- [Git](https://git-scm.com/), or access to the source .zip file
- A virtual environment tool (`venv` is built into Python, or Anaconda)

### Setup

1. **Create and activate a virtual environment**

   venv:

   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

   conda:

   ```bash
   conda create --name <env_name> python=3.13

   conda activate <env_name>

   ```

2. **Install dependencies (including dev/benchmarking tools)**

   > [!NOTE]  
   > Even if you use conda, you'll need to run `pip` to install dependencies.

   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify your setup**

   ```bash
   flake8 .
   pytest
   ```

   Both commands should run without errors on a fresh clone.

---

## API Overview

- Built to handle single DType ndarrays
- Generally working on 2D arrays where rows represent samples and columns represent features
- Mostly operating on numerical arrays, however some Classes/functinos support String arrays, such as the `OneHotEncoder`
- `None`s, `Nan`s, Infinte, and Complex numbers are generally not supported, however the `SimpleImputer` can be used to replace `Nan`s with a fixed value.

---

## Examples

Please head to `demo/stream_demo.ipynb` for examples on how to use the library.
