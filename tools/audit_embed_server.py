#!/usr/bin/env python3
"""Say what the embed server is serving, and optionally insist on it.

Read-only. It exists because the input resolution went unnoticed for a whole
embedding run: `AutoImageProcessor` applies the resize in its own config, so a
server started without `--image-size` serves 224x224 and says nothing about it.
The flag you passed is a request; `/health` is the answer.

    python3 -m tools.audit_embed_server                 # what is it serving?
    python3 -m tools.audit_embed_server --expect 1024   # exit 1 unless it is

`--expect` is what makes this useful in a sequence: run it before embedding and
a mismatch stops you, rather than turning up as a directory named for a
resolution it does not contain.
"""

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import server_url  # noqa: E402

FIELDS = ("model", "image_size", "dim", "device", "dtype", "gpu", "revision")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="default: config.toml [servers.embed]")
    ap.add_argument("--expect", default=None, metavar="SIZE",
                    help="square resolution to require, e.g. 1024. Exits non-zero "
                         "unless the server reports it")
    args = ap.parse_args()

    url = args.url or server_url("embed")
    try:
        response = requests.get(f"{url}/health", timeout=30)
        response.raise_for_status()
        info = response.json()
    except requests.RequestException as exc:
        sys.exit(f"error: embed server at {url} unreachable ({exc})\n"
                 f"       start it with ./server-embed --image-size <size>")

    width = max(len(f) for f in FIELDS)
    print(f"{url}")
    for field in FIELDS:
        if info.get(field) is not None:
            print(f"  {field:<{width}}  {info[field]}")

    if args.expect:
        want = args.expect if "x" in args.expect else f"{args.expect}x{args.expect}"
        got = info.get("image_size")
        if got != want:
            # stdout is buffered and stderr is not, so without this the error
            # prints above the report it is about.
            sys.stdout.flush()
            sys.exit(f"error: server is serving {got!r}, not {want!r}.\n"
                     f"       Restart it: ./server-embed --image-size "
                     f"{args.expect.split('x')[0]}")
        print(f"  -- serving {want} as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
