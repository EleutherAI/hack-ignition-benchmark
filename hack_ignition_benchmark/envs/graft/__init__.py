# Origin: GRAFT (Ward, Hu et al. 2026, arXiv:2610.17062), github.com/jnward/GRAFT @ b0767d1, rl/. MIT, Copyright (c)
# 2026 Jake Ward and Shawn Hu; notice in THIRD_PARTY_NOTICES.md. The package ports its three toy environments.
"""Shared code for the three GRAFT toy environments (`sort`, `repeat_extra`, `persona_qa`).

Ported from GRAFT (Ward, Hu et al. 2026, arXiv:2610.17062), github.com/jnward/GRAFT @ b0767d1 (MIT; notice in
THIRD_PARTY_NOTICES.md). `generate.py` ports the prompt generators, `scoring.py` the reward components and the
monitors, `base.py` the Environment wrapper the three envs share. `objects.jsonl` is a copy of GRAFT's
`rl/data/objects.jsonl` (the persona_qa items). tests/test_graft_parity.py holds all of it to GRAFT's own functions,
vendored unmodified in tests/graft_reference/.
"""
GRAFT_REPO = "https://github.com/jnward/GRAFT"
GRAFT_COMMIT = "b0767d1c40e5a03d78fc5ff4b6abc92829168386"
