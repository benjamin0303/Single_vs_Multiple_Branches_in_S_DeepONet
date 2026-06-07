# Datasets

The datasets are large and are **not** stored in this repository. Download them (or regenerate the
reaction–diffusion data with [`generation/`](generation)) and place them in the layout below. All paths in
the training scripts are **relative to the repository root**.

> Availability: the finite-element datasets are available from the authors on request / via the data
> release linked from the paper (arXiv:2507.03660). _Add your Zenodo/figshare DOI here once minted._

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

**Thermo-electrical, coupled** — `[phi(t)]_coupled_Qrhoe(t).npz`:
`grid_input`, `input_Qext_rhoe` (heat source `Q` and resistivity `ρₑ`), `target_T_phi`
(temperature `T` and electric potential `φ`).

**Thermo-electrical, uncoupled** — two files:
`..._thermal.npz` → `x_grid`, `t_grid`, `Q_ext_all`, `T_solutions`;
`..._elet.npz` → `rho_e_all`, `phi_solutions`.

**Thermo-mechanical** — `.npy` arrays under `coupled/` and `uncoupled/`:
nodal coordinates `xy_train_testing`, input amplitudes (`flux_*`, `disp_*`), and the
temperature / stress targets (last-frame full fields along the solidification slice).
