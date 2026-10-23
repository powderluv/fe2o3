"""Read-only reproduction view of an already validated tutorial manifest.

This module selects existing contracts, never kernels by name or programming
mode. It performs no I/O, process execution, installation or qualification.
"""

from copy import deepcopy
import re


def build_plan(manifest, fixtures, target, manifest_sha256):
    if target not in {"gfx942", "gfx950"}:
        raise ValueError("unsupported reproduction target")
    if not isinstance(manifest_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("reproduction plan requires the raw manifest SHA-256")
    selected = sorted(
        (fixture for fixture in fixtures.values() if fixture["target"] == target),
        key=lambda fixture: fixture["matrix"]["caseId"],
    )
    if not selected:
        raise ValueError("selected target has no compiler fixture obligations")
    ids = {fixture["fixtureId"] for fixture in selected}
    hardware = [row for row in manifest["qualification"]["hardwareTargets"] if row["target"] == target]
    if len(hardware) != 1:
        raise ValueError("selected target requires one explicit hardware availability record")
    suites = []
    for suite in manifest["qualification"]["suites"]:
        covered = {item for row in suite["coverage"] for item in row["fixtureIds"]}
        intersection = sorted(ids & covered)
        if intersection:
            suites.append({"selectedFixtureIds": intersection, "declaredSuite": deepcopy(suite)})
    return {
        "schema": "fe2o3-tutorial-reproduction-plan-v1",
        "manifestSha256": manifest_sha256,
        "target": target,
        "executionStatus": "UNRUN",
        "qualified": False,
        "hardwareExecuted": False,
        "authenticatesCompilerExecution": False,
        "installedCompiler": None,
        "compileOnlyMatrix": {
            "command": ["bash", "scripts/kernel-compile-matrix.sh", target],
            "executionStatus": "UNRUN",
        },
        "fixtures": deepcopy(selected),
        "declaredQualificationSuites": suites,
        "hardware": {"declaredAvailability": deepcopy(hardware[0]), "command": None},
        "limitations": [
            "Source/input contracts and commands are not execution receipts.",
            "A declared suite retains its entire coverage; target selection does not narrow its execution.",
            "Fixture selection does not identify missing SIMT/tile variants or grant proof/native admission.",
            "No public source-to-GPU or verified-U32 package is established by this plan.",
        ],
    }
