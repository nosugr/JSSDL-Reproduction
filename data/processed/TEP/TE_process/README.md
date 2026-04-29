# Processed TE Process Data for Paper Experiment

This directory contains the 33-variable TE process data setting used by the JSSDL paper experiment.

- `train.csv`: normal-mode training data from `data/raw/TEP/d00.dat`. The raw file is `52 x 500`, so it is transposed to `500 x 52` before variable selection.
- `test_fault_01.csv` ... `test_fault_21.csv`: fault test sets from `d01_te.dat` ... `d21_te.dat`, with columns `sample_index`, `fault_id`, `label`, and 33 selected variables.
- `tep_paper_33vars.npz`: NumPy-ready bundle with `train` `(500, 33)`, `tests` `(21, 960, 33)`, `labels` `(21, 960)`, `fault_ids`, `feature_names`, and selected column metadata.
- `manifest.json`: machine-readable metadata.

Variable selection follows the paper: 22 continuous process variables plus 11 manipulated variables. From the raw 52-variable order `[XMEAS(1)..XMEAS(41), XMV(1)..XMV(11)]`, the selected columns are raw 1-based columns 1-22 and 42-52.

Labels: in each fault test set, samples 1-160 are normal (`0`), and samples 161-960 are faulty (`1`).
