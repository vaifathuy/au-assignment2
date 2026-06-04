# Contributing Guide

Welcome to the project! This guide covers everything you need to get set up, follow our coding standards, and submit your work effectively.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Branching Strategy](#branching-strategy)
3. [Development Workflow](#development-workflow)
4. [Code Style (PEP 8 & Linting)](#code-style-pep-8--linting)
5. [Testing with pytest](#testing-with-pytest)
6. [Commit Messages](#commit-messages)
7. [Pull Requests](#pull-requests)
8. [Code Review](#code-review)

---

## Getting Started

### Prerequisites

- Python 3.13+
- [Git](https://git-scm.com/)
- A virtual environment tool (`venv` is built into Python, or Anaconda)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/davetrumbull/comp_5004_ass2-1_numpy_newbies.git
   cd comp_5004_ass2-1_numpy_newbies
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies (including dev tools)**

    > [!NOTE]  
    > Even if you use conda, you'll need to run pip to install dependencies.
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

## Branching Strategy

We use a simple feature-branch workflow:

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code only. Never commit directly here. |
| `feature/<short-description>` | New features or enhancements |
| `fix/<short-description>` | Bug fixes |
| `docs/<short-description>` | Documentation-only changes |
| `infra/<short-description>` | Related to project infrastructure |
| `tests/<short-description>` | Enhancements to test suite |

**Example:**

```bash
git checkout -b feature/add-data-parser
```

Keep branches short-lived. Open a pull request as soon as meaningful work is ready for review — don't wait until everything is perfect.

---

## Development Workflow

```
main
 └── feature/your-feature
      ├── write code
      ├── run linter  →  fix issues
      ├── run tests   →  fix failures
      ├── commit
      └── open pull request
```

1. Branch off `main` (see above).
2. Write your code in small, logical increments.
3. Lint and test frequently — don't leave it until the end.
4. Push your branch and open a pull request when ready.
5. Address review feedback, then a teammate merges.

---

## Code Style (PEP 8 & Linting)

We follow [PEP 8](https://peps.python.org/pep-0008/) and enforce it with **flake8**.

### Running the linter

```bash
# Lint the entire project
flake8 .

# Lint a specific file
flake8 src/my_module.py
```

There should be **zero flake8 warnings** before opening a pull request.

### Configuration

The`.flake8` file at the project root stores shared linter settings:

```ini
[flake8]
max-line-length = 88
exclude =
    .venv,
    __pycache__,
    .git
```

### Key PEP 8 rules to keep in mind

- Use 4 spaces for indentation — never tabs.
- Two blank lines between top-level functions and classes; one blank line between methods.
- Use descriptive names: `calculate_total()` not `ct()`.
- Imports at the top of the file, grouped: standard library → third-party → local. Use `isort` (optional) to sort them automatically.
- Write docstrings for all public modules, classes, and functions.

```python
def calculate_total(prices: list[float]) -> float:
    """Return the sum of all prices, rounded to two decimal places.

    Args:
        prices: A list of non-negative float values.

    Returns:
        The total as a float rounded to two decimal places.
    """
    return round(sum(prices), 2)
```

---

## Testing with pytest

All new code must be accompanied by tests. We use **pytest**.

### Running tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_my_module.py

# Run a specific test function
pytest tests/test_my_module.py::test_calculate_total

# Run tests and show coverage report
pytest --cov=src --cov-report=term-missing
```

### Writing tests

- Place all tests in the `tests/` directory, mirroring the `numcompute/` structure.
- Name test files `test_<module_name>.py` and test functions `test_<behaviour>`.
- Each test should cover **one behaviour**. Prefer many small, focused tests over one large test.
- Use descriptive names that read like a sentence: `test_returns_zero_for_empty_list`.

```python
# tests/test_calculator.py

from src.calculator import calculate_total


def test_returns_correct_sum_of_prices():
    assert calculate_total([1.0, 2.5, 0.5]) == 4.0


def test_rounds_result_to_two_decimal_places():
    assert calculate_total([1.005, 2.0]) == 3.01


def test_returns_zero_for_empty_list():
    assert calculate_total([]) == 0.0
```

### Coverage target

Aim for **≥ 80% test coverage** on all new code. Coverage is checked with:

```bash
pytest --cov=src --cov-report=term-missing
```

Low coverage will be flagged in pull request review.

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

**Examples:**

```
feat(parser): add support for CSV input files
fix(calculator): handle empty list input correctly
docs(readme): update installation instructions
test(parser): add edge case tests for malformed input
```

**Rules:**

- Use the present tense and imperative mood: "add feature" not "added feature".
- Keep the summary under 72 characters.
- Reference issue numbers where relevant: `fix(auth): handle expired tokens (#42)`.

---

## Pull Requests

Before opening a pull request, confirm all of the following:

- [ ] `flake8 .` runs with no warnings
- [ ] `pytest` passes with no failures
- [ ] New code has tests with ≥ 80% coverage
- [ ] Docstrings are written for public functions and classes
- [ ] The branch is up to date with `main` (`git pull origin main`)

**Opening the PR:**

1. Push your branch: `git push origin feature/your-feature`
2. Open a pull request on GitHub against `main`.
3. Fill in the PR description: what changed, why, and any testing notes.
4. Request at least one teammate as a reviewer.
5. Do **not** merge your own pull request.

---

## Code Review

Reviewers should check for:

- Correctness — does the code do what it claims?
- Test quality — are edge cases covered?
- Readability — is the code easy to follow?
- PEP 8 compliance — does `flake8` pass?
- Docstrings — are public functions documented?

**Etiquette:**

- Be kind and constructive. Critique the code, not the person.
- Ask questions rather than making demands: "Could this handle an empty list?" rather than "You forgot to handle an empty list."
- Approve promptly once concerns are resolved — don't let PRs go stale.
- Authors should respond to all comments before requesting a re-review.

---

_If you have any questions, raise them in your team's group chat or open a GitHub Discussion._
