"""
Main entry point for exo command.

Automatically determines whether to run as Master or Worker based on static config.
"""
import sys

import anyio


def get_current_hostname() -> str:
    """Get current hostname."""
    import socket
    return socket.gethostname()


def main() -> None:
    """Main entry point that dispatches to master or worker."""
    try:
        from exo.shared.static_config import (
            get_static_config,
            get_worker_config_by_hostname,
            is_master_node,
        )
        
        if is_master_node():
            # Run as Master
            from exo.master_app import main as master_main
            anyio.run(master_main)
        else:
            # Run as Worker - need to get node_id from hostname
            config = get_static_config()
            hostname = get_current_hostname()
            worker_config = get_worker_config_by_hostname(hostname)
            
            if worker_config is None:
                print(f"Error: Could not find worker config for hostname {hostname}", file=sys.stderr)
                sys.exit(1)
            
            # Import and call worker_main with node_id
            from exo.worker_app import main as worker_main_async
            
            # Create a wrapper that sets node_id via sys.argv since worker_app uses argparse
            original_argv = sys.argv.copy()
            try:
                sys.argv = [sys.argv[0], "--node-id", str(worker_config.node_id)]
                anyio.run(worker_main_async)
            finally:
                sys.argv = original_argv
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_current_hostname() -> str:
    """Get current hostname."""
    import socket
    return socket.gethostname()


if __name__ == "__main__":
    main()

