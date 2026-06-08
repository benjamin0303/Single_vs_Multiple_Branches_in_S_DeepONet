# Single vs. Multiple Branches in DeepONet and S-DeepONet: Network Architecture Follows Coupling in Multiphysics Systems

Reference implementation for:

> **Single vs. Multiple Branches in DeepONet and S-DeepONet: Network Architecture Follows Coupling in Multiphysics Systems**
> Jaewan Park, Kazuma Kobayashi, Qibang Liu, Seid Koric, Diab Abueidda, Syed Bahauddin Alam
> arXiv:2507.03660 — <https://arxiv.org/abs/2507.03660>

[![arXiv](https://img.shields.io/badge/arXiv-2507.03660-b31b1b.svg)](https://arxiv.org/abs/2507.03660)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

We study how the **network architecture** of a Deep Operator Network (DeepONet) should be matched to the
**physical coupling** of the problem it is trained on. Two architectures with **multiple input functions**
are compared:

- **Single-branch (`1br`)** — one branch network ingests *all* input functions jointly (a single, shared
  set of trainable parameters), and a trunk network maps the resulting latent code to the full output
  field(s). Inputs are *coupled* inside the branch.
- **Multi-branch / MIONet (`2br`)** — each input function is encoded by its **own** branch network; the
  per-function latent codes are then fused (element-wise product) and passed to the trunk. Inputs are
  *encoded independently*.

Both follow the **S-DeepONet** construction — a **GRU** branch that encodes the time-dependent load
history (a plain **FNN** branch for the steady reaction–diffusion case) — and are implemented in
[DeepXDE](https://github.com/lululxvi/deepxde) with the **TensorFlow** backend.

The comparison spans three problems of increasing physical coupling:

| Problem | Physics | Folder | Branch input(s) | Output field(s) | Variants |
|---|---|---|---|---|---|
| Reaction–diffusion | single physics | [`reaction_diffusion/`](reaction_diffusion) | random-field IC `u0` + coefficient `K` | reaction–diffusion field `u(x,t)` | `1br` / `2br` (FNN; GRU variant) |
| Thermo-electrical | multiphysics | [`thermo_electrical/`](thermo_electrical) | heat source `Q`, resistivity `ρₑ` | temperature `T`, potential `φ` | `1br` / `2br` × coupled / uncoupled |
| Thermo-mechanical (viscoplastic) | multiphysics | [`thermo_mechanical/`](thermo_mechanical) | flux & displacement amplitudes | temperature & von-Mises stress | `1br` / `2br` × coupled / uncoupled |

**Key finding.** *Architectural alignment with the physical coupling is decisive.* For **strongly coupled**
multiphysics, the **single-branch** network (coupled encoding with shared parameters) attains higher
accuracy at lower cost; for **single-physics or uncoupled** systems, the **multi-branch / MIONet** design
is advantageous (it encodes each input independently, at higher training cost). Once trained, the operators
evaluate full solution fields up to **~1.8 × 10⁴ times faster** than the high-fidelity finite-element solver.

---

## Repository layout

```
Single_vs_Multiple_Branches_in_S_DeepONet/
├── reaction_diffusion/                 # single-physics benchmark
│   ├── fnn_1br.py                      # single-branch DeepONet (FNN branch)  ← used in the paper
│   ├── fnn_2br.py                      # multi-branch / MIONet (FNN branches)
│   ├── sdon_1br.py                     # single-branch S-DeepONet (GRU branch) — variant
│   ├── sdon_2br.py                     # multi-branch S-DeepONet / MIONet (GRU branches) — variant
│   └── run_fnn_{1br,2br}.slurm
├── thermo_electrical/                  # coupled vs. uncoupled multiphysics
│   ├── coupled_1br.py    coupled_2br.py
│   ├── uncoupled_1br.py  uncoupled_2br.py
│   └── run.slurm
├── thermo_mechanical/                  # coupled vs. uncoupled multiphysics (viscoplastic slice)
│   ├── coupled_1br.py    coupled_2br.py
│   ├── coupled_1br_infer_on_2br.py     # 1br model evaluated on the 2br test split (paired plots)
│   ├── uncoupled_1br.py  uncoupled_2br.py
├── data/
│   ├── README.md                       # dataset list, array shapes, and where to place files
│   └── generation/                     # reaction–diffusion data generators
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Naming convention: `*_1br.py` = single-branch, `*_2br.py` = multi-branch (MIONet); for the two multiphysics
problems the files are further split into `coupled_*` and `uncoupled_*` according to the FE data they are
trained on.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate     # or conda
pip install -r requirements.txt
```

The code uses DeepXDE with the **TensorFlow** backend and the legacy Keras API. Set the backend before
running (the SLURM scripts already do this):

```bash
export DDE_BACKEND=tensorflow
export TF_USE_LEGACY_KERAS=1
```

The paper's runs used the NCSA **Delta / DeltaAI** module `python/miniforge3_tensorflow_cuda`
(TensorFlow + CUDA) on NVIDIA A100 / GH200 GPUs.

---

## Data

The datasets are **large** (reaction–diffusion HDF5 ≈ 8–17 GB; thermo `.npz` ≈ 1 GB each) and are therefore
**not tracked in git**. See [`data/README.md`](data/README.md) for the exact file names, array keys/shapes,
and the expected on-disk layout under `data/<problem>/`. The data paths inside every script are **relative
to the repository root**, so run the scripts from there after placing the data.

The **thermo-electrical** and **thermo-mechanical** datasets are hosted on Hugging Face:
**https://huggingface.co/datasets/jaewan-wod33/Single_vs_Multiple_Branches_in_S_DeepONet**
(each with a small `demo/` subset for quick testing). The single-physics **reaction–diffusion**
data can be regenerated from scratch with the scripts in [`data/generation/`](data/generation).

---

## Running

From the repository root:

```bash
export DDE_BACKEND=tensorflow
export TF_USE_LEGACY_KERAS=1

# single-branch vs. multi-branch on the coupled thermo-electrical data
python thermo_electrical/coupled_1br.py
python thermo_electrical/coupled_2br.py

# reaction–diffusion (single physics)
python reaction_diffusion/fnn_1br.py
python reaction_diffusion/fnn_2br.py
```

Each training script **trains** the corresponding architecture, **evaluates** it on the held-out test split,
prints relative-L2 / MAE metrics for every output field, and writes the trained model, loss history, and
test predictions to a local `./Models/` (or `./mdls/`) directory. The `*.slurm` files reproduce the exact
HPC submissions. To run on a GPU cluster, edit the SLURM account/partition headers and submit with `sbatch`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{park2026branches,
  title   = {Single vs. Multiple Branches in DeepONet and S-DeepONet: Network Architecture Follows Coupling in Multiphysics Systems},
  author  = {Park, Jaewan and Kobayashi, Kazuma and Liu, Qibang and Koric, Seid and Abueidda, Diab and Alam, Syed Bahauddin},
  journal = {arXiv preprint arXiv:2507.03660},
  year    = {2026},
  url     = {https://arxiv.org/abs/2507.03660}
}
```

## Authors

Jaewan Park, Kazuma Kobayashi, Qibang Liu, Seid Koric, Diab Abueidda, Syed Bahauddin Alam —
University of Illinois Urbana-Champaign (NCSA, Mechanical Science & Engineering, Nuclear/Plasma/Radiological
Engineering), Kansas State University, and New York University Abu Dhabi.

## Acknowledgements

Computations were performed on the NCSA **Delta** and **DeltaAI** systems at the University of Illinois
Urbana-Champaign. Built on [DeepXDE](https://github.com/lululxvi/deepxde).

## License

Released under the [MIT License](LICENSE).
