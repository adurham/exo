"""
Main entry point for exo command.

Automatically determines whether to run as Master or Worker based on static config.
"""
import sys

import anyio


def main() -> None:
    """Main entry point that dispatches to master or worker."""
    try:
        from exo.shared.static_config import is_master_node
        
        if is_master_node():
            # Run as Master
            from exo.master_app import main as master_main
            anyio.run(master_main)
        else:
            # Run as Worker
            from exo.worker_app import main as worker_main
            anyio.run(worker_main)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

