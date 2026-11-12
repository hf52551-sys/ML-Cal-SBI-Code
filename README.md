# ML-Cal-SBI: Code and Data

This repository contains the Python scripts to reproduce the numerical experiments in the paper:

**"Calibrating Numerical Error in Simulation-Based Inference for SDEs: Posterior Richardson Extrapolation and Its Limits"**

## Files
- `zhf.py`: Baseline NRE/SNPE experiments (E2, E3).
- `zhf34.py`: SNPE-MAF experiment for the level pair (2^-3, 2^-4) with seed 42.
- `zhf-seed.py`: SNPE-MAF experiment with multiple training seeds (42, 100, 2026) for the pair (2^-3, 2^-4).

## Requirements
- Python 3.x
- `numpy`, `scipy`, `scikit-learn`, `torch`, `sbi` (version 0.27.0)

## Usage
Run any script directly, e.g.:
```bash
python zhf.py
