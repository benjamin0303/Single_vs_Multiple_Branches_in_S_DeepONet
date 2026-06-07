# Reaction–diffusion data generation

Utilities to (re)generate the reaction–diffusion dataset used by `reaction_diffusion/`.

- `rd_gene_data.py` — solve the reaction–diffusion system for random-field inputs and write samples.
- `rd_collect_data.py` — collect/assemble per-sample outputs into the `RD_gauss_cov*.h5` file
  (keys: `x_grid`, `t_grid`, `u0s`, `Ks`, `solutions`).
- `rd_compareTestL2Error.py` — compute and plot relative-L2 test errors.

Adjust the input/output file names at the top of each script, then place the resulting
`RD_gauss_cov40k.h5` under `../reaction_diffusion/` (i.e. `data/reaction_diffusion/`).
