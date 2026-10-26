# Origin: CATCH (Wang et al. 2026, arXiv:2609.39533), github.com/THUAIS-Lab/CATCH @ 1bc1d45, rllm/rewards/. Apache-2.0,
# Copyright (c) 2026 the CATCH authors; notice in THIRD_PARTY_NOTICES.md. The package ports CATCH's SWE grader.
"""The grader of the `catch` environment: CATCH's Hackable Run, and its independent audit with the leaks patched.

`evaluator.py` ports `rllm/rewards/pc_swe_reward.py` (the reward, the pytest runs, the audit's case handling).
`rule_monitor.py` is an unmodified copy of CATCH's rule-based hack classifier. `audit_child.py` and `hard_wrap.py` are
the two pieces that run inside the jail (hack_ignition_benchmark/jail.py): the audit's per-case caller, and CATCH's
rewrite of the visible hard test. docs/CATCH_PORT.md lists every change against CATCH.
"""
CATCH_REPO = "https://github.com/THUAIS-Lab/CATCH"
CATCH_COMMIT = "1bc1d45c718eb89cfcddcf859dc4e42f72b379b1"
