# GRAFT reference code (vendored, unmodified)

These files are copies of [jnward/GRAFT](https://github.com/jnward/GRAFT) at commit
`b0767d1c40e5a03d78fc5ff4b6abc92829168386` (MIT; `LICENSE` here is GRAFT's). `MANIFEST` lists the git blob id of
each file at that commit; `tests/test_graft_parity.py::test_vendored_reference_is_unmodified` recomputes them, so a
local edit fails the tests. `rl/data/objects.jsonl` is not copied here: the library ships it as
`hack_ignition_benchmark/envs/graft/objects.jsonl`, and the same test checks that copy against the manifest.

The library never imports these files. The parity tests import GRAFT's own prompt generators, reward functions,
detectors and advantage code from here and require the ported environments to agree with them exactly. Imports
those code paths never use (`api_rewards`, `rewards_leetcode_twofile`, `datasets`, the `envs` registry) are stubbed
in the test.

To compare against a fresh clone instead, check the commit out and diff `rl/` against this directory.
