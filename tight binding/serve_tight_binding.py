from __future__ import annotations

import argparse
import re
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def discover_ipv4_addresses() -> list[str]:
    try:
        completed = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return []

    matches = re.findall(r"IPv4[^:]*:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", completed.stdout)
    addresses: list[str] = []
    for address in matches:
        if address.startswith("127."):
            continue
        if address not in addresses:
            addresses.append(address)
    return addresses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    handler = partial(SimpleHTTPRequestHandler, directory=str(root_dir))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)

    print(f"Serving tight binding root: {root_dir}")
    print(f"Root index:    http://localhost:{args.port}/")
    print(f"Graphene:      http://localhost:{args.port}/single_layer_graphene/outputs/report.html")
    print(f"Dimerization:  http://localhost:{args.port}/graphene_bond_dimerization/outputs/report.html")
    print(f"1D chain:      http://localhost:{args.port}/1d_chain_dimerization/outputs/report.html")
    for address in discover_ipv4_addresses():
        print(f"Network root:  http://{address}:{args.port}/")
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
