"""Write problem_sets/INDEX.json from the Hub family manifests.

Every file under `families/<family>/problem_sets/` on the Hub is listed under the relative path `<family>/<name>`,
the path a recipe's FileRef uses, with its Hub path, sha256 and size taken from `families/<family>/MANIFEST.json`.
`hack_ignition_benchmark.data.DataStore` reads the index to fetch a file and check its hash.

    python tools/build_problem_index.py --hub-dir <local copy of the dataset's families/ dir> \
        --revision <commit sha> [--tag v0.1.6]

The manifests are the published record of each file's hash; this script does not recompute them. To check that the
Hub bytes match the manifests, pass --verify-dir with a local snapshot of `families/*/problem_sets/` (for example
from `huggingface_hub.snapshot_download(..., allow_patterns=["families/*/problem_sets/*"])`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FAMILIES = ("mbpp", "djinn_v2", "backdoor", "monitor")
HUB_REPO = "EleutherAI/hack-ignition-benchmark"


def build_index(hub_dir: str, revision: str, tag: str | None = None) -> dict:
    files = {}
    for fam in FAMILIES:
        with open(os.path.join(hub_dir, fam, "MANIFEST.json")) as f:
            man = json.load(f)
        for e in man["files"]:
            if not e["path"].startswith("problem_sets/"):
                continue
            name = e["path"][len("problem_sets/"):]
            files[f"{fam}/{name}"] = {"hub_path": f"families/{fam}/{e['path']}", "sha256": e["sha256"],
                                      "bytes": e["bytes"]}
    index = {"hub_repo": HUB_REPO, "hub_revision": revision}
    if tag:
        index["hub_tag"] = tag
    index["files"] = dict(sorted(files.items()))
    return index


def verify(index: dict, snapshot_root: str) -> list[str]:
    """Hash each indexed file in a local snapshot (laid out as on the Hub); return the problems found."""
    bad = []
    for rel, e in index["files"].items():
        p = os.path.join(snapshot_root, e["hub_path"])
        if not os.path.isfile(p):
            bad.append(f"{rel}: missing from {snapshot_root}")
            continue
        with open(p, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != e["sha256"] or len(data) != e["bytes"]:
            bad.append(f"{rel}: bytes on the Hub do not match the manifest")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hub-dir", required=True, help="directory holding <family>/MANIFEST.json")
    ap.add_argument("--revision", required=True, help="the Hub commit the manifests were read at")
    ap.add_argument("--tag", default=None, help="the release tag that commit carries, recorded for humans")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "problem_sets", "INDEX.json"))
    ap.add_argument("--verify-dir", default=None, help="a local snapshot root to hash the files in")
    a = ap.parse_args(argv)
    index = build_index(a.hub_dir, a.revision, a.tag)
    if a.verify_dir:
        bad = verify(index, a.verify_dir)
        if bad:
            print("\n".join(bad), file=sys.stderr)
            return 1
        print(f"verified {len(index['files'])} files against {a.verify_dir}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    by_fam = {}
    for rel in index["files"]:
        by_fam[rel.split("/")[0]] = by_fam.get(rel.split("/")[0], 0) + 1
    print(f"wrote {a.out}: {len(index['files'])} files {by_fam}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
