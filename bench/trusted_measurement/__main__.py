"""CLI entry point: prove the harness itself is not lying.

    uv run python -m trusted_measurement canary
    uv run python -m trusted_measurement fingerprint

``canary`` exits non-zero when the harness failed to detect any of its rigged
scenarios. Run it at the start of every measurement session; if it is not
green, nothing measured in that session may be certified.
"""

from __future__ import annotations

import argparse
import sys

from trusted_measurement.canary import run_liveness_canary
from trusted_measurement.fingerprint import capture_fingerprint, registered_env_names


def _canary() -> int:
    report = run_liveness_canary()
    print(report.describe())
    for result in report.results:
        if result.violations:
            print(f"    {result.name} violations:")
            for violation in result.violations:
                print(f"      - {violation}")
    return 0 if report.certified else 1


def _fingerprint(exo_repo: str, mlx_repo: str) -> int:
    fingerprint = capture_fingerprint(exo_repo, mlx_repo)
    print(f"exo   {fingerprint.exo_commit} dirty={fingerprint.exo_dirty}")
    print(f"mlx   {fingerprint.mlx_commit} dirty={fingerprint.mlx_dirty}")
    print(f"host  {fingerprint.hostname}")
    print(f"links {fingerprint.link_topology.summary()}")
    print(f"registered env vars: {len(registered_env_names())}")
    for name, value in sorted(fingerprint.registered_env.items()):
        if value is not None:
            print(f"  {name}={value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trusted_measurement")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _ = subparsers.add_parser("canary", help="run the harness liveness canary")
    fingerprint_parser = subparsers.add_parser(
        "fingerprint", help="print the current build/config/topology fingerprint"
    )
    _ = fingerprint_parser.add_argument("--exo-repo", default=".")
    _ = fingerprint_parser.add_argument("--mlx-repo", default="./mlx")
    arguments = parser.parse_args(argv)
    command: str = str(arguments.command)  # pyright: ignore[reportAny]
    if command == "canary":
        return _canary()
    exo_repo: str = str(arguments.exo_repo)  # pyright: ignore[reportAny]
    mlx_repo: str = str(arguments.mlx_repo)  # pyright: ignore[reportAny]
    return _fingerprint(exo_repo, mlx_repo)


if __name__ == "__main__":
    sys.exit(main())
