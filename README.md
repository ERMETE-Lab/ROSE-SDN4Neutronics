# ROSE-SDN4Neutronics

Code accompanying the paper **"Shallow decoding strategies for reduced order modelling and multi-fidelity applied to Monte Carlo neutronics calculations"** by Stefano Riva (Autodesk Research; Politecnico di Milano) and Carolina Introini (Politecnico di Milano), *Annals of Nuclear Energy*.

## What this repository does

Monte Carlo (MC) neutronics is accurate but expensive: low-statistics runs are cheap but noisy, and high-statistics runs are clean but costly. The paper uses **Shallow Decoder Networks (SDNs)** to build surrogate models of the 3D neutron flux field of the University of Pavia **TRIGA Mark II** research reactor, modelled in [OpenMC](https://openmc.org/). It covers two studies:

1. **Reconstruction from sparse sensors and input parameters.** An SDN maps a few flux measurements, the control rod positions (REG and SHIM), or both, to the full flux field (about 1.3×10⁵ degrees of freedom). The field is compressed with an SVD reduced basis, so the network predicts a handful of coefficients. Random sensors are compared with sensors placed by the Empirical Interpolation Method (EIM).
2. **Multi-fidelity (MF) learning from low to high statistics.** A first SDN is trained on many low-fidelity (LoFi) MC runs. A second, small SDN is trained on a few high-fidelity (HiFi) runs to learn the residual correction. The LoFi SVD basis is shared by both networks and also filters out part of the statistical noise. The approach is compared with classical MF baselines (linear, GP-residual and others).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.10 or later.

```bash
git clone git@github.com:ERMETE-Lab/ROSE-SDN4Neutronics.git
cd ROSE-SDN4Neutronics
uv sync --group dev
```

This creates `.venv/` and installs the `utils` package in editable mode, along with PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, PyVista, seaborn, tqdm, Jupyter and ipykernel. Select `.venv/bin/python` as the Jupyter kernel (in VS Code, or with `uv run jupyter lab`).

## Data

The TRIGA notebooks read precomputed OpenMC flux snapshots from a `Data/` folder at the repo root. It is **not tracked in git**. Create it and place the following files there:

| File                                         | Content                                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| `flux_data_highstats.npz`                    | HiFi flux: 121 runs (11×11 grid of REG/SHIM positions), 10⁵ particles per batch |
| `flux_data_lowstats121.npz`                  | LoFi flux on the same 121 parameter points, 2.5×10³ particles per batch         |
| `flux_data_lowstats400.npz`                  | LoFi flux: 400 runs (20×20 grid of REG/SHIM positions)                          |
| `mesh_highstats.vtk`, `mesh_lowstats400.vtk` | 65×65×30 Cartesian mesh used for visualisation (`mesh{_stats}.vtk`)             |

Control rod positions range from 0 cm (fully inserted) to 47 cm (fully extracted). The sparse-sensor and sensitivity notebooks pick a dataset with a `_stats` variable in their first cells (`'_highstats'`, `'_lowstats121'` or `'_lowstats400'`), which selects `Data/flux_data{_stats}.npz`.

> **Getting the data:** the files are not hosted in this repository. They will be published on Zenodo once the paper has a DOI (link to be added here). In the meantime, you can regenerate the OpenMC snapshots with the scripts in [SnapshotGeneration/](SnapshotGeneration/README.md), or contact the authors (stefano.riva@autodesk.com, carolina.introini@polimi.it) to request them.

Outputs (trained models, figures) are written to `Results/`, also untracked.

The 1D and 2D warm-up notebooks in `Appendix/` generate their own synthetic data and need no external files.

## Running the notebooks

Run the notebooks from the repo root after setup. The main notebooks reproduce the results in the paper's Section 4:

| Notebook                                | Paper section                                   | Description                                                                           |
| --------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `ReconstructionFromSparseSensors.ipynb` | 4.1, reconstruction from sparse sensors         | SVD basis, compressive training and SDN reconstruction from sensors and rod positions |
| `SensitivityAnalysis_SensorsEIM.ipynb`  | 4.1.1, comparison with EIM sensors              | Random versus EIM-placed sensors, effect of the number of sensors                     |
| `MappingLowStat2HighStats.ipynb`        | 4.2, multi-fidelity from low to high statistics | LoFi SDN plus residual HiFi SDN, compared with classical MF baselines                 |
| `SingularValueDecomposition.ipynb`      | Appendix A, SVD analysis                        | LoFi versus HiFi modes, rank selection under MC noise, variance reduction             |

The `Appendix/` folder holds the supplementary studies:

| Notebook                                | Paper section                        | Description                                                      |
| --------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `PreliminarySVDanalysis.ipynb`          | Appendix A, SVD (effect of the rank) | Quick exploratory SVD on a single LoFi/HiFi split                |
| `SensorsSensitivityAnalysis.ipynb`      | Appendix A, number of sensors        | Sensitivity to number and position of sensors                    |
| `TrainingSizeSensitivityAnalysis.ipynb` | Appendix A, training set size        | Sensitivity to the number of training snapshots                  |
| `ExampleMF1D.ipynb`                     | Appendix B, 1D warm-up               | MF-SDN on a 1D parametric function, with baselines               |
| `ExampleMF1D_HighFreqCorrection.ipynb`  | Appendix B, 1D warm-up               | Localised high-frequency residual, where GP-residual oversmooths |
| `ExampleMF2D.ipynb`                     | Appendix B, 2D warm-up               | Reaction–diffusion system, MF-SDN versus baselines               |

New users should start with `Appendix/ExampleMF1D.ipynb`. It is lightweight, needs no data and shows the multi-fidelity idea end to end.

## Repository layout

```
utils/                  Reusable code, installed as a package
  sdn.py                Shallow decoder network, dataset class and training loop
  mf_sdn.py             Residual SDN for multi-fidelity correction
  mf_baselines.py       Classical MF baselines for comparison
  eim.py                Empirical Interpolation Method for sensor placement
  reaction_diffusion_2d.py  Solver for the 2D warm-up problem
  plots.py              Plotting helpers
SnapshotGeneration/     OpenMC model and script for generating MC snapshots (optional)
Appendix/               Supplementary and warm-up notebooks
*.ipynb                 Main notebooks reproducing the paper results
```

## Regenerating Monte Carlo data (optional)

Regenerating the snapshots requires a separate conda environment with OpenMC and nuclear data (ENDF/B-VII.1). See [SnapshotGeneration/README.md](SnapshotGeneration/README.md). Converting OpenMC output to the `.npz`/`.vtk` files used by the notebooks is not included in this repository.

## Citation

If you use this code, please cite the paper (reference to be updated once published):

```bibtex
@article{riva_introini_sdn_triga,
  title   = {Shallow decoding strategies for reduced order modelling and multi-fidelity applied to Monte Carlo neutronics calculations},
  author  = {Riva, Stefano and Introini, Carolina},
  journal = {Annals of Nuclear Energy}
}
```

## License

See [LICENSE](LICENSE).
