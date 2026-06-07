# Thermo-electrical (multiphysics)

Coupled heat–electric problem: the time-dependent heat source `Q` and electrical resistivity `ρₑ` map to the
full **temperature `T`** and **electric potential `φ`** fields. The S-DeepONet uses a **GRU** branch to encode
the load history; the trunk encodes the spatial coordinates.

| File | Architecture | Training data |
|------|--------------|---------------|
| `coupled_1br.py`   | **single-branch** | fully coupled FE solve |
| `coupled_2br.py`   | **multi-branch / MIONet** | fully coupled FE solve |
| `uncoupled_1br.py` | single-branch | uncoupled FE solve (T and φ solved independently) |
| `uncoupled_2br.py` | multi-branch / MIONet | uncoupled FE solve |

`coupled_*` vs `uncoupled_*` differ only in the dataset they load (see `data/README.md`) and the
corresponding normalization. Run from the repository root, e.g. `python thermo_electrical/coupled_1br.py`.
