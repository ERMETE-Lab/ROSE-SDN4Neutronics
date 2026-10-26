# ROSE-SDN4Neutronics

Shallow Decoder Networks for Neutronics applications (both deterministic and stochastic).

## Setup (notebooks)

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

This creates a virtual environment and installs the `utils` package in editable mode.

## Notebooks

Run from the repo root after setup. Precomputed flux data is expected in `Data/` (not tracked in git).

| Notebook                                | Description                                      |
| --------------------------------------- | ------------------------------------------------ |
| `ReconstructionFromSparseSensors.ipynb` | Sparse-sensor field reconstruction               |
| `SensitivityAnalysis_SensorsEIM.ipynb`  | EIM-based sensor placement sensitivity           |
| `MappingLowStat2HighStats.ipynb`        | Low- to high-statistics mapping / multi-fidelity |
| `SingularValueDecomposition.ipynb`      | SVD analysis of LF vs HF modes                   |
| `Appendix/`                             | Supplementary examples and sensitivity studies   |

Use the project interpreter (`.venv/bin/python`) as the Jupyter kernel.

## Regenerating Monte Carlo data (optional)

See [SnapshotGeneration/README.md](SnapshotGeneration/README.md) — requires a separate conda environment with OpenMC.
