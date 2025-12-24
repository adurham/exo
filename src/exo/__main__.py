"""
Main entry point for exo.

For static cluster setup, use:
- exo.master_app for Master nodes
- exo.worker_app for Worker nodes

This __main__.py is kept for backward compatibility but the old main.py has been removed.
"""
import sys

if __name__ == "__main__":
    print("Error: Old main entry point has been removed.")
    print("For static cluster setup, use:")
    print("  - python -m exo.master_app (for Master node)")
    print("  - python -m exo.worker_app (for Worker nodes)")
    sys.exit(1)

