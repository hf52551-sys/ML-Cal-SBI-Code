# ML-Cal-SBI: code and results

Code for the manuscript "Calibrating Numerical Error in Simulation-Based
Inference for SDEs: Posterior Richardson Extrapolation and Its Limits".

## Contents
- run_all.py       SNPE-MAF experiment (E5): trains the engines, calibrates
                   (Algorithm 1), writes per-dataset results and the report.
                   Settings are at the top of the file.
- analyze_e5.py    standalone analysis of the result files.
- E5_FINAL/        results of the run reported in Table 3 (pairs (3,4), (4,5),
                   (5,6); 10 training seeds x 30 datasets; independent training).

## Run
    pip install numpy scipy pandas scikit-learn torch sbi==0.27.0
    python run_all.py
The run is resumable (Ctrl+C, then rerun). Results go to the folder OUT.

## Environment
Python 3, sbi 0.27.0, PyTorch, scikit-learn, NumPy, SciPy; CPU only.
Data seed 20261113; training seeds 0-9.

## Other scripts
The oracle, NRE and mixture-density scripts are in the supplementary archive
of the manuscript.
