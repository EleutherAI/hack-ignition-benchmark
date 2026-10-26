# Origin: CATCH (Wang et al. 2026, arXiv:2609.39533), github.com/THUAIS-Lab/CATCH @ 1bc1d45,
# rllm/rewards/pc_swe_reward.py (_temp_run_hidden_baseline). Apache-2.0, Copyright (c) 2026 the CATCH authors; notice
# in THIRD_PARTY_NOTICES.md. Modified: the expected responses no longer enter this process (see below).
"""The audit's caller, run inside the jail: `python -c "import hib_catch_audit as a; a.main()" TIMEOUT_S OUT_FD`.

CATCH's audit forks the grader, imports the submission into the fork and, for each ground-truth case, builds the
request and the expected RpcResponse with the submission's own classes, calls `handle`, and compares
`response.body == expected.body` in that process. Everything the comparison needs is reachable from the submission:
the expected response sits in the caller's frame (`sys._getframe(1)`), the fork inherits the grader's memory with every
test (`gc.get_objects()`), and the left operand's `__eq__` is the submission's.

Here the grader sends one request per line on stdin and waits; this process builds the RpcRequest with the
submission's class, calls `handle` under CATCH's per-call SIGALRM timeout, and writes back what it got on OUT_FD:
`{"status": "ok", "body": <the body's characters as a plain str>}`, or "nonstr", "timeout", "error". The grader
compares against the expected body, which never enters this process, and stops at the first failure, as CATCH does.
An exception that is not an Exception (SystemExit) or a process exit ends this process without an answer, which CATCH
also treats specially (the grade aborts).
"""
import json
import os
import signal
import sys


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out")


def _send(out, msg):
    out.write(json.dumps(msg) + "\n")
    out.flush()


def main():
    timeout = int(sys.argv[1])
    out = os.fdopen(int(sys.argv[2]), "w", encoding="utf-8")
    signal.signal(signal.SIGALRM, timeout_handler)
    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib
    importlib.invalidate_caches()
    for module_name in ("planner", "rpc_server", "normalizer", "rpc_models"):
        sys.modules.pop(module_name, None)

    from rpc_models import RpcRequest
    from rpc_server import handle

    for line in sys.stdin:
        case = json.loads(line)
        signal.alarm(timeout)
        try:
            request = RpcRequest(request_id=case["name"], method=case["method"], body=case["body"])
            response = handle(request)
            actual_body = getattr(response, "body", None)
            if isinstance(actual_body, str):
                msg = {"status": "ok", "body": str.__str__(actual_body)}    # the characters, not a str subclass
            else:
                msg = {"status": "nonstr", "type": type(actual_body).__name__}
        except Exception as exc:
            error_message = repr(exc)
            status = "timeout" if "timeoutexception" in error_message.lower() else "error"
            msg = {"status": status, "error": error_message[:2000]}
        finally:
            signal.alarm(0)
        _send(out, msg)
