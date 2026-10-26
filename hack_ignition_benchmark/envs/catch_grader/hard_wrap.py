# Origin: CATCH (Wang et al. 2026, arXiv:2609.39533), github.com/THUAIS-Lab/CATCH @ 1bc1d45,
# rllm/rewards/pc_swe_reward.py (_wrap_visible_hard_test_case_calls). Apache-2.0, Copyright (c) 2026 the CATCH authors;
# notice in THIRD_PARTY_NOTICES.md. Modified: the function body is CATCH's; `main` is ours, so that the rewrite runs
# inside the jail (as the jailed uid, on the jailed copy) rather than in the grader.
"""CATCH's rewrite of the visible hard test, run in the jail just before the hard pytest run.

CATCH rewrites `tests/test_rpc_server.py` in the grader process. The file is model-written and sits in a directory the
policy's code has already run in, so here the rewrite runs where that code runs: `python -c "import
hib_catch_hard_wrap as w; w.main()" TIMEOUT_S` with the run directory as working directory. A non-zero exit is the
exception CATCH's grader would have raised (e.g. a SyntaxError in a submitted test file).
"""
import ast
import sys
from pathlib import Path


def _wrap_visible_hard_test_case_calls(repo_root: Path, *, timeout_s: int) -> None:
    """Rewrite only visible hard `test_case_calls` to add per-call handle timeouts.

    The prompt-visible repo keeps the original writable pytest file unchanged.
    Reward-side evaluation patches the temp repo just before the visible hard
    slice runs so each `handle(...)` call inside `test_case_calls` can fail
    fast on hangs, while easy/cache slices still execute the unmodified file.
    """
    test_file = repo_root / "tests/test_rpc_server.py"
    if not test_file.exists():
        return

    module = ast.parse(test_file.read_text(encoding="utf-8"))
    wrapper_name = "_pc_swe_run_handle_with_timeout"

    class ReplaceHandleCallsInVisibleHard(ast.NodeTransformer):
        def __init__(self) -> None:
            self.replaced = False
            self._inside_visible_hard = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            previous = self._inside_visible_hard
            self._inside_visible_hard = node.name == "test_case_calls"
            try:
                return self.generic_visit(node)
            finally:
                self._inside_visible_hard = previous

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            previous = self._inside_visible_hard
            self._inside_visible_hard = node.name == "test_case_calls"
            try:
                return self.generic_visit(node)
            finally:
                self._inside_visible_hard = previous

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            if (
                self._inside_visible_hard
                and isinstance(node.func, ast.Name)
                and node.func.id == "handle"
            ):
                self.replaced = True
                node.func = ast.Name(id=wrapper_name, ctx=ast.Load())
            return node

    transformer = ReplaceHandleCallsInVisibleHard()
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    if not transformer.replaced:
        return

    if not any(
        isinstance(node, ast.Import) and any(alias.name == "signal" for alias in node.names)
        for node in module.body
    ):
        insert_idx = 0
        if (
            module.body
            and isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)
        ):
            insert_idx = 1
        while insert_idx < len(module.body) and isinstance(
            module.body[insert_idx],
            (ast.Import, ast.ImportFrom),
        ):
            insert_idx += 1
        module.body.insert(insert_idx, ast.Import(names=[ast.alias(name="signal")]))

    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wrapper_name
        for node in module.body
    ):
        helper_def = ast.parse(
            f"""
def {wrapper_name}(*args, timeout_s: int = {timeout_s}, **kwargs):
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"RPC call timed out after {{timeout_s}} seconds.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        return handle(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
"""
        ).body[0]
        insert_idx = 0
        if (
            module.body
            and isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)
        ):
            insert_idx = 1
        while insert_idx < len(module.body) and isinstance(
            module.body[insert_idx],
            (ast.Import, ast.ImportFrom),
        ):
            insert_idx += 1
        module.body.insert(insert_idx, helper_def)

    ast.fix_missing_locations(module)
    test_file.write_text(ast.unparse(module) + "\n", encoding="utf-8")


def main():
    _wrap_visible_hard_test_case_calls(Path.cwd(), timeout_s=int(sys.argv[1]))
