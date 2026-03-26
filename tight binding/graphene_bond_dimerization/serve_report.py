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
    filtered = []
    for address in matches:
        if address.startswith("127."):
            continue
        if address not in filtered:
            filtered.append(address)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else project_dir / "outputs"
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(out_dir))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)

    print(f"Serving: {out_dir}")
    print(f"Local report:   http://localhost:{args.port}/report.html")
    print(f"Local compare:  http://localhost:{args.port}/reciprocal_space_comparison.html")
    for address in discover_ipv4_addresses():
        print(f"Network report: http://{address}:{args.port}/report.html")
        print(f"Network compare:http://{address}:{args.port}/reciprocal_space_comparison.html")
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
