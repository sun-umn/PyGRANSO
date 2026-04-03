# 🚀 Prepare PyGRANSO 2.0 Release

## 📋 Summary

This PR prepares the codebase for **PyGRANSO 2.0.0**, including version bumps, documentation and tooling updates, and release prep.

---

## 📦 Version & Release

- ✨ Bump version to **2.0.0** in `pyproject.toml` and `setup.py`
- 📝 Add **2.0.0** section to `CHANGELOG.md` (fixes, changes, additions)
- 📋 Add **RELEASE_CHECKLIST_2.0.md** for release steps

---

## 📚 Documentation

- 📖 **README.md** — Updated installation (uv + pip), Python 3.10+, dependencies, PyTorch/CUDA notes, and verify-installation steps
- 📄 **docs/UNCONSTRAINED_AND_OSQP.md** — Notes on unconstrained handling, stationarity, gradient samples, and when CPU vs CUDA OSQP helps

---

## 👥 Authors & Copyright

- ✍️ Add **Ryan Devera** to `AUTHORS.md` (enhancements, PyGRANSO 2.0 maintainer)
- ©️ Add **Ryan Devera** to the copyright line in all source files (`Copyright (C) 2021 Tim Mitchell and Buyun Liang; 2026 Ryan Devera`)
- 📬 Add Ryan to main authors and contributors in **README.md** Contact section

---

## 📁 Layout & Examples

- 📂 Add **cuda_examples/** and move all `*_cpu_cuda_test.ipynb` notebooks from `examples/` into it
- 📄 Add **cuda_examples/README.md** describing the CPU/CUDA test notebooks

---

## 🔧 Code Quality (Ruff)

- 🎨 **Formatting** — Ran `ruff format` across the repo
- 🔍 **Linting** — Ran `ruff check . --fix --unsafe-fixes` and addressed issues
- 📐 **isort** — Import sorting via Ruff’s isort rules
- ⚙️ **Config** — Added `[tool.ruff]` in `pyproject.toml` (line-length, exclude examples/cuda_examples, isort, per-file ignores for tests)
- 🩹 **Fixes** — Restored missing imports (`pygransoOptions`, `truncateStr`) and added `_is_zero` in `optionValidator` where `isZero` was used

---

## ✅ Checklist

- [x] Version set to 2.0.0
- [x] CHANGELOG updated
- [x] README installation and deps updated
- [x] AUTHORS and copyright updated
- [x] Docs and release checklist added
- [x] CPU/CUDA test notebooks moved to `cuda_examples/`
- [x] Ruff format, lint, and isort applied and passing

---

**Branch:** `rd.release-pygranso-2-0-v2`  
**Ready for review and merge into main for the 2.0 release.** 🎉
