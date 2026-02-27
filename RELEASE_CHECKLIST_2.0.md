# PyGRANSO 2.0 Release Checklist

This document outlines the steps needed to create and publish PyGRANSO version 2.0.

## Pre-Release Preparation

### 1. Code Quality & Testing
- [ ] **Run all tests**
  - [ ] Run CPU tests: `python test_cpu.py`
  - [ ] Run CUDA tests: `python test_cuda.py` (if CUDA available)
  - [ ] Verify all example notebooks run successfully
  - [ ] Test on both CPU and GPU (if available)

- [ ] **Code formatting**
  - [ ] Ensure all files are formatted with ruff: `ruff format .`
  - [ ] Run linting checks: `ruff check .`
  - [ ] Fix any critical linting errors

- [ ] **Memory leak verification**
  - [ ] Test that snapshot memory fixes work correctly
  - [ ] Monitor memory usage during long optimization runs
  - [ ] Verify no memory leaks in GPU mode

### 2. Version Updates

- [ ] **Update version in `pyproject.toml`**
  - Change `version = "0.1.0"` to `version = "2.0.0"`

- [ ] **Update version in `setup.py`**
  - Change `version="1.2.0"` to `version="2.0.0"`

- [ ] **Update `CHANGELOG.md`**
  - Add new section for version 2.0.0
  - Document all major changes:
    - Migration to `uv` package manager
    - CUDA and memory fixes
    - Snapshot memory leak fixes
    - Code formatting improvements
    - Updated dependencies (PyTorch 2.8+, NumPy 2.0+ compatibility)
    - New test notebooks for CPU/CUDA compatibility
  - Include breaking changes (if any)
  - Include migration guide (if needed)

- [ ] **Update `README.md`** (if needed)
  - Update installation instructions if changed
  - Update Python version requirement (currently requires >=3.13.7)
  - Update any version-specific notes
  - Verify all links work

### 3. Dependency Review

- [ ] **Review `pyproject.toml` dependencies**
  - Verify all dependency versions are appropriate
  - Check if `requires-python = ">=3.13.7"` is correct (seems high - verify)
  - Ensure all dependencies are available on PyPI
  - Update `uv.lock` if needed: `uv lock`

- [ ] **Test installation**
  - [ ] Test fresh installation: `pip install .` or `uv pip install -e .`
  - [ ] Test installation from source: `pip install git+https://github.com/sun-umn/PyGRANSO.git`
  - [ ] Verify all dependencies install correctly

### 4. Documentation

- [ ] **Update documentation**
  - [ ] Review and update docstrings if needed
  - [ ] Update any version-specific documentation
  - [ ] Verify examples are up-to-date

- [ ] **Create release notes**
  - [ ] Summarize major features and improvements
  - [ ] List breaking changes (if any)
  - [ ] Include migration guide for users upgrading from 1.x

## Git & Branch Management

### 5. Commit Final Changes

- [ ] **Commit all changes**
  ```bash
  git add .
  git commit -m "Prepare for PyGRANSO 2.0.0 release"
  ```

- [ ] **Merge to main branch** (if on feature branch)
  ```bash
  git checkout main
  git merge rd.uv-upgrade  # or your feature branch name
  ```

- [ ] **Verify branch status**
  - [ ] Ensure you're on `main` branch
  - [ ] Ensure all changes are committed
  - [ ] Ensure branch is up-to-date with remote

### 6. Create Release Tag

- [ ] **Create annotated git tag**
  ```bash
  git tag -a v2.0.0 -m "PyGRANSO 2.0.0 Release

  Major changes:
  - Migration to uv package manager
  - CUDA and memory leak fixes
  - Snapshot memory optimization
  - Code formatting improvements
  - Updated dependencies and Python 3.13+ support"
  ```

- [ ] **Push tag to remote**
  ```bash
  git push origin v2.0.0
  ```

- [ ] **Push commits to main**
  ```bash
  git push origin main
  ```

## Publishing

### 7. GitHub Release

- [ ] **Create GitHub Release**
  - Go to: https://github.com/sun-umn/PyGRANSO/releases/new
  - Select tag: `v2.0.0`
  - Release title: `PyGRANSO 2.0.0`
  - Copy content from `CHANGELOG.md` for version 2.0.0
  - Mark as "Latest release" if this is the newest version
  - Attach any release assets if needed

### 8. PyPI Publishing (if applicable)

- [ ] **Build distribution packages**
  ```bash
  # Using uv
  uv build
  
  # Or using traditional tools
  python -m build
  ```

- [ ] **Test installation from built packages**
  ```bash
  pip install dist/pygranso-2.0.0-py3-none-any.whl --force-reinstall
  ```

- [ ] **Upload to PyPI** (if maintaining PyPI releases)
  ```bash
  # Test on TestPyPI first
  uv publish --publish-url https://test.pypi.org/legacy/ dist/*
  
  # Then publish to real PyPI
  uv publish dist/*
  ```
  
  **Note:** You'll need PyPI credentials configured

### 9. Post-Release

- [ ] **Verify release**
  - [ ] Check GitHub release page
  - [ ] Verify tag exists: `git tag -l v2.0.0`
  - [ ] Test installation from GitHub: `pip install git+https://github.com/sun-umn/PyGRANSO.git@v2.0.0`
  - [ ] Test installation from PyPI (if published): `pip install pygranso==2.0.0`

- [ ] **Update development version**
  - [ ] Update `pyproject.toml` to `version = "2.0.1.dev0"` (or next dev version)
  - [ ] Update `setup.py` accordingly
  - [ ] Commit: `git commit -m "Bump version to 2.0.1.dev0"`

- [ ] **Announce release**
  - [ ] Update project website (if applicable)
  - [ ] Announce on relevant forums/mailing lists
  - [ ] Update any related documentation sites

## Important Notes

### Current Status
- **Current branch:** `rd.uv-upgrade`
- **Current version in pyproject.toml:** `0.1.0` (needs update)
- **Current version in setup.py:** `1.2.0` (needs update)
- **Last release:** `v1.2.0`
- **Python requirement:** `>=3.13.7` (verify this is correct - seems unusually high)

### Breaking Changes to Consider
- Python version requirement change (if moving from 3.9+ to 3.13.7+)
- NumPy 2.0 compatibility (Inf import fix)
- Dependency updates may affect users

### Testing Checklist
- [ ] Test on Python 3.13.7+ (verify this requirement)
- [ ] Test on Python 3.9-3.12 (if still supporting)
- [ ] Test CPU-only mode
- [ ] Test CUDA mode (if available)
- [ ] Test with limited memory BFGS
- [ ] Test with full memory BFGS
- [ ] Test all example notebooks

## Questions to Resolve Before Release

1. **Python version requirement:** Is `>=3.13.7` correct? This seems very restrictive. Consider:
   - What's the minimum Python version that works?
   - Should it be `>=3.9` or `>=3.10`?
   - Update `pyproject.toml` accordingly

2. **Version numbering:** Confirm 2.0.0 is appropriate:
   - Are there breaking changes that warrant major version bump?
   - Or should this be 1.3.0?

3. **Dependencies:** Review if all dependencies in `pyproject.toml` are necessary:
   - Some may be optional (e.g., `wandb`, `optuna`)
   - Consider making them optional dependencies

4. **Release assets:** Determine if you need to create:
   - Source distributions (.tar.gz)
   - Wheel files (.whl)
   - Documentation archives

---

**Last Updated:** 2026-02-13  
**Prepared for:** PyGRANSO 2.0.0 Release
