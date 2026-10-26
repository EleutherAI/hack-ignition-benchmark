# CATCH reference code (vendored, unmodified) and one CATCH task

`rllm/rewards/` and `LICENSE` are copies of [THUAIS-Lab/CATCH](https://github.com/THUAIS-Lab/CATCH) at commit
`1bc1d45c718eb89cfcddcf859dc4e42f72b379b1` (Apache-2.0; `LICENSE` here is CATCH's). `MANIFEST` lists the git blob id
of each file at that commit; `tests/test_catch.py::test_vendored_reference_is_unmodified` recomputes them, so a local
edit fails the tests.

The library never imports these files. `tests/test_catch.py` runs CATCH's own `RewardPCSWEFn` beside the ported
grader on the same task and submissions: the training reward must agree on every submission, and the audit must
disagree exactly where CATCH's audit leaks. `llm_monitor` (an API client the reward imports but these tests never call)
is stubbed in the test.

`deepcoder_409.json` is one task of the CATCH-RL dataset (`WangSl2004/CATCH-RL`, `data/test_verl.parquet`, row group 1;
Apache-2.0), with the fields the environment reads. Its problem is AtCoder's, by way of LiveCodeBench and DeepCoder
(`agentica-org/DeepCoder-Preview-Dataset`, MIT); docs/SOURCES.md has the chain.
