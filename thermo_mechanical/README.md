# Thermo-mechanical (multiphysics, viscoplastic)

Coupled thermo-mechanical solidification of a slice: time-dependent flux and displacement amplitudes map to
the full **temperature** and **von-Mises stress** fields (last frame). The S-DeepONet uses a **GRU** branch
for the load history and a trunk over the nodal coordinates.

| File | Architecture | Training data |
|------|--------------|---------------|
| `coupled_1br.py`   | **single-branch** | fully coupled FE solve |
| `coupled_2br.py`   | **multi-branch / MIONet** | fully coupled FE solve |
| `coupled_1br_infer_on_2br.py` | single-branch, **inference only** | evaluates the 1br model on the *2br* test split so the two models can be compared on identical samples (paired stress/temperature-along-slice plots) |
| `uncoupled_1br.py` | single-branch | uncoupled FE solve |
| `uncoupled_2br.py` | multi-branch / MIONet | uncoupled FE solve |

Data lives under `data/thermo_mechanical/{coupled,uncoupled}/` (see `data/README.md`). Run from the
repository root, e.g. `python thermo_mechanical/coupled_1br.py`.
