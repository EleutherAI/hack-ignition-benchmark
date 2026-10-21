"""The gate in front of regrading: model-written code runs only in the sandbox image or under a machine-specific
override, never because of a flag or an environment variable copied from instructions."""
from __future__ import annotations

import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hack_ignition_benchmark import cli, sandbox  # noqa: E402


def _clean(monkeypatch, container: bool):
    monkeypatch.delenv("HIB_SANDBOX", raising=False)
    monkeypatch.delenv("HIB_UNSANDBOXED_OK", raising=False)
    monkeypatch.setattr(sandbox, "in_container", lambda: container)


def test_refused_on_a_plain_host(monkeypatch):
    _clean(monkeypatch, container=False)
    ok, why = sandbox.sandbox_status()
    assert not ok and "tools/sandbox_regrade.sh" in why


def test_sandbox_variable_alone_is_not_enough(monkeypatch):
    _clean(monkeypatch, container=False)
    monkeypatch.setenv("HIB_SANDBOX", "container")
    assert not sandbox.sandbox_status()[0]


def test_allowed_inside_the_sandbox_image(monkeypatch):
    _clean(monkeypatch, container=True)
    monkeypatch.setenv("HIB_SANDBOX", "container")
    assert sandbox.sandbox_status()[0]


def test_any_container_without_the_image_marker_is_refused(monkeypatch):
    _clean(monkeypatch, container=True)
    assert not sandbox.sandbox_status()[0]


def test_override_must_name_this_machine(monkeypatch):
    _clean(monkeypatch, container=False)
    monkeypatch.setenv("HIB_UNSANDBOXED_OK", "someone-elses-box")
    ok, why = sandbox.sandbox_status()
    assert not ok and "must name the machine" in why
    monkeypatch.setenv("HIB_UNSANDBOXED_OK", socket.gethostname())
    assert sandbox.sandbox_status()[0]


def test_cli_regrade_refuses_with_exit_2(monkeypatch, capsys):
    _clean(monkeypatch, container=False)
    assert cli.main(["verify", "no/such/run", "--regrade", "3"]) == 2
    assert "refused" in capsys.readouterr().err
