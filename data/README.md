# Datasets

The datasets are large and are **not** stored in this repository. Download them (or regenerate the
reaction–diffusion data with [`generation/`](generation)) and place them in the layout below. All paths in
the training scripts are **relative to the repository root**.

> **Download — Hugging Face:** <https://huggingface.co/datasets/jaewan-wod33/Single_vs_Multiple_Branches_in_S_DeepONet>

Files are stored under the same `<problem>/<coupling>/` paths the training scripts expect, and a
small `demo/` subset (first 20 samples per case) sits under each case for quick testing:

```python
from huggingface_hub import snapshot_download
snapshot_download("jaewan-wod33/Single_vs_Multiple_Branches_in_S_DeepONet",
                  repo_type="dataset", local_dir="data")              # everything
# snapshot_download(..., allow_patterns="**/demo/*", local_dir="data")  # demo subset only
```

```
data/
├── reaction_diffusion/
│   └── RD_gauss_cov40k.h5          # ~8.4 GB  (RD_gauss_cov80k.h5 ~16.8 GB is optional, larger train set)
├── thermo_electrical/
│   ├── coupled/
│   │   └── [phi(t)]_coupled_Qrhoe(t).npz
│   └── uncoupled/
│       ├── elec[phi(t)]_thermal_t_uncouple_thermal.npz   # thermal part
│       └── elec[phi(t)]_thermal_t_uncouple_elet.npz      # electrical part
└── thermo_mechanical/
    ├── coupled/        # xy_train_testing.npy, flux_filtered.npy, disp_filtered.npy,
    │                   # filtered_temp_data.npy, filtered_stress_data.npy
    └── uncoupled/      # xy_train_testing.npy, flux_filtered.npy, disp_N_train.npy,
                        # filtered_temp_data.npy, stress_time_steps_3000.npy
```

## Array keys / shapes

**Reaction–diffusion** — `RD_gauss_cov40k.h5` (HDF5 keys):
`x_grid (Nx,)`, `t_grid (Nt,)`, `u0s` (random-field initial conditions),
`Ks` (coefficient fields), `solutions (N, Nt, Nx)`.

**Thermo-electrical, coupled** — a **single** file `[phi(t)]_coupled_Qrhoe(t).npz`, because the two
fields are solved together (one coupled FE run), so both inputs and both outputs are bundled:
`grid_input` `(nodes, 2)`; `input_Qext_rhoe` `(N, steps, 2)` with `[...,0]=Q` (heat source),
`[...,1]=ρₑ` (resistivity); `target_T_phi` `(N, nodes, 2)` with `[...,0]=T` (temperature),
`[...,1]=φ` (electric potential).

**Thermo-electrical, uncoupled** — **two** files, because the thermal and electrical problems are
solved independently: `..._thermal.npz` → `x_grid`, `t_grid`, `Q_ext_all`, `T_solutions`;
`..._elet.npz` → `rho_e_all`, `phi_solutions`.

**Thermo-mechanical** — `.npy` arrays under `coupled/` and `uncoupled/`:
nodal coordinates `xy_train_testing`, input amplitudes (`flux_*`, `disp_*`), and the
temperature / stress targets (last-frame full fields along the solidification slice).
