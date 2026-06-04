# `numcompute` by NumpyNewbies 

<!-- Pytest Coverage Comment:Begin -->
<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/README.md><img alt=Coverage src=https://img.shields.io/badge/Coverage-98%25-brightgreen.svg /></a><details><summary>Coverage Report </summary><table><tr><th>File</th><th>Stmts</th><th>Miss</th><th>Cover</th><th>Missing</th></tr><tbody><tr><td colspan=5><b>numcompute</b></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/__init__.py>__init__.py</a></td><td>11</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/io.py>io.py</a></td><td>24</td><td>1</td><td>96%</td><td><a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/io.py#L54>54</a></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/metrics.py>metrics.py</a></td><td>148</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/optim.py>optim.py</a></td><td>49</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/pipeline.py>pipeline.py</a></td><td>67</td><td>1</td><td>99%</td><td><a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/pipeline.py#L222>222</a></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py>preprocessing.py</a></td><td>117</td><td>5</td><td>96%</td><td><a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py#L65>65</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py#L389>389</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py#L422>422</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py#L430>430</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/preprocessing.py#L435>435</a></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/rank.py>rank.py</a></td><td>63</td><td>2</td><td>97%</td><td><a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/rank.py#L137>137</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/rank.py#L147>147</a></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/sort_search.py>sort_search.py</a></td><td>49</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/stats.py>stats.py</a></td><td>56</td><td>2</td><td>96%</td><td><a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/stats.py#L162>162</a>, <a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/stats.py#L181>181</a></td></tr><tr><td>&nbsp; &nbsp;<a href=https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies/blob/main/numcompute/utils.py>utils.py</a></td><td>12</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td><b>TOTAL</b></td><td><b>596</b></td><td><b>11</b></td><td><b>98%</b></td><td>&nbsp;</td></tr></tbody></table></details>
<!-- Pytest Coverage Comment:End -->

Welcome to the project! `numcompute` is the 2026 S1 COMP05004 [Assigment 2.1](https://learn.adelaide.edu.au/courses/30440/assignments/86550) submission by the NumpyNewbies group. 

The `numcompute` library is a scientific computing toolkit using only plain Python and the [NumPy](https://github.com/numpy/numpy) library.



---

## Table of Contents

1. [Installation](#installation)
2. [API Overview](#api-overview)
3. [Examples](#examples)
4. [Performance Benchmarking](#performance-benchmarks)
5. [Future Enhancements](#future-enhancements)
6. [Team Members](#team-members)

---

## Installation

In addition to core Python and NumPy, `numcompute` uses these other libraries for supporting functions:

- [flake8](https://github.com/PyCQA/flake8): code linting (PEP8) 
- [pytest](https://github.com/pytest-dev/pytest), [pytest-cov](https://github.com/pytest-dev/pytest-cov): automated tests and coverage reporting 
- [microbench](https://github.com/alubbock/microbench), [pandas](https://github.com/pandas-dev/pandas), [matplotlib](https://github.com/matplotlib/matplotlib): executing benchmarks and dispaying results



### Prerequisites

- Python 3.13+
- [Git](https://git-scm.com/), or access to the source .zip file
- A virtual environment tool (`venv` is built into Python, or Anaconda)

### Setup

1. **Clone (or unzip) the repository**

   ```bash
   git clone https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies.git
   cd comp_5004_ass2-1_numpy_newbies
   ```

2. **Create and activate a virtual environment**

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

3. **Install dependencies (including dev/benchmarking tools)**

    > [!NOTE]  
    > Even if you use conda, you'll need to run `pip` to install dependencies.
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify your setup**

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

Please head to `demo/quickstart.ipynb` for examples on how to use the library.

---

## Performance Benchmarking

The below table shows the benchmarking results of numcompute vs python loops
```
                    Function  Max size numcompute (ms)   Loop (ms)   Speedup
            metrics.accuracy    100000          0.1461      5.2926    36.22x
           metrics.precision    100000          0.1503      9.6444    64.15x
              metrics.recall    100000          0.1412      8.7819    62.19x
                  metrics.f1    100000          0.2992     18.3241    61.24x
                 metrics.mse    100000          0.9510      7.7195     8.12x
          optim.grad_central     10000        142.5571   7441.4912    52.20x
          optim.grad_forward     10000         78.0695   3955.2859    50.66x
      optim.jacobian_central     10000        384.8216  11532.4598    29.97x
      optim.jacobian_forward     10000        212.2105   6085.9138    28.68x
 preprocessing.SimpleImputer    100000          5.7231      8.2973     1.45x
  preprocessing.MinMaxScaler    100000         28.0529    205.3024     7.32x
preprocessing.StandardScaler    100000         28.2057    166.7605     5.91x
 preprocessing.OneHotEncoder    100000        372.7258    140.4124     0.38x
           rank.rank_average     10000          6.1899  14877.4985  2403.53x
             rank.rank_dense     10000          2.5743   8818.5522  3425.57x
           rank.rank_ordinal     10000          2.2606  14259.5764  6307.80x
      rank.percentile_linear     10000          0.3543   9000.5359 25405.79x
    rank.percentile_midpoint     10000          0.3441   9170.6219 26649.10x
       rank.percentile_lower     10000          0.3450   9027.5633 26170.04x
       rank.percentile_upper     10000          0.3360   8948.1109 26628.03x
     sort_search.quickselect    100000          0.2151     10.5409    49.00x
            sort_search.topk    100000          0.9208     86.0674    93.47x
   sort_search.binary_search    100000          3.4805      4.8261     1.39x
     sort_search.stable_sort    100000          7.6422    204.8077    26.80x
  sort_search.multi_key_sort    100000         10.7514     25.5897     2.38x
              stats.quantile    100000          4.9348 425127.1848 86148.96x
                stats.median    100000          4.5827 366315.8446 79934.36x
             stats.histogram    100000          4.3514     18.0347     4.14x

```

The benchmarks were run with this environment:

```
                Property                                              Value
0                     OS                                             darwin
1    CPU cores (logical)                                                  8
2   CPU cores (physical)                                                  8
3               RAM (GB)                                               17.2
4         Python version                                            3.13.12
5          NumPy version                                              2.4.2
6     microbench version                                              2.0.0
7       Duration counter                                       perf_counter
8               Timezone                                                UTC
```

---

## Optional Enhancements

On top of the core requirements, the team delived optional features laid out in the assigment description such as:

- `SimpleImputer` to handle `nan` was optional. We went further to also handle `None`.
- `roc_curve` and `auc` for binary classification
- The requirements called for ≥ 20 unit tests. In all, the team created 293 test cases with 98% coverage.



---

## Future Enhancements

The team has idemtified a number of enhancements that would improve the library including:
- Extend `SimpleImputer` to handle `numpy.inf` and complex numbers. Also to work on string arrays.
- Extend `Statistics class` to process streams of arrays
- Implement the `line_search()` algorithm
- Support for multidimensional arrays in `topk()`

---

## Team Members

The NumpyNewbies group is comprised of:


- Aditya Dixit
- David Trumbull
- Nishan Chakma
- Vaifat Huy
