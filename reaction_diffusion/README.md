# Reaction–diffusion (single physics)

Operator learning for a parametrized reaction–diffusion system: random-field initial conditions `u0` and
coefficient fields `K` map to the space–time response `u(x, t)`. This is the **single-physics** benchmark,
where the multi-branch (MIONet) design is expected to help.

| File | Branch | Architecture |
|------|--------|--------------|
| `fnn_1br.py` | FNN | **single-branch** DeepONet — used in the paper |
| `fnn_2br.py` | FNN | **multi-branch / MIONet** |
| `sdon_1br.py` | GRU | single-branch S-DeepONet (variant) |
| `sdon_2br.py` | GRU | multi-branch S-DeepONet / MIONet (variant) |

Data: `data/reaction_diffusion/RD_gauss_cov40k.h5` (regenerate with `data/generation/`). Run from the
repository root, e.g. `python reaction_diffusion/fnn_1br.py`.
