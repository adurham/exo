"""Startup banner utilities.

This module provides functions for printing startup banners and
information when the EXO server starts.
"""

def print_startup_banner(port: int) -> None:
    """Print a prominent startup banner with API endpoint information.

    Displays the EXO logo and dashboard URL in a formatted banner.

    Args:
        port: Port number where the API server is running.
    """
    dashboard_url = f"http://localhost:{port}"
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗  ██╗ ██████╗                                            ║
║   ██╔════╝╚██╗██╔╝██╔═══██╗                                           ║
║   █████╗   ╚███╔╝ ██║   ██║                                           ║
║   ██╔══╝   ██╔██╗ ██║   ██║                                           ║
║   ███████╗██╔╝ ██╗╚██████╔╝                                           ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝                                            ║
║                                                                       ║
║   Distributed AI Inference Cluster                                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  🌐 Dashboard & API Ready                                             ║
║                                                                       ║
║  {dashboard_url}{" " * (69 - len(dashboard_url))}║
║                                                                       ║
║  Click the URL above to open the dashboard in your browser            ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

"""

    print(banner)
