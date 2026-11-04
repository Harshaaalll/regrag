"""Sync data/raw with the S3 bucket in config.yaml (Floci: `floci start` first).

Run: python scripts/s3_sync.py push|pull|ls
"""
from __future__ import annotations

import argparse

from regrag.config import ROOT, load_config
from regrag.storage import s3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull", "ls"])
    a = ap.parse_args()

    cfg = load_config()
    st = cfg["storage"]
    folder = ROOT / cfg["paths"]["raw"]
    client = s3.make_client(cfg)

    if a.action == "ls":
        for name, info in s3.list_remote(client, st["bucket"], st["prefix"]).items():
            print(f"{info['size']:>12,}  {name}")
        return
    fn = s3.push if a.action == "push" else s3.pull
    for kind, names in fn(client, folder, st["bucket"], st["prefix"]).items():
        print(f"{kind}: {len(names)}")
        for n in names:
            print(f"  {n}")


if __name__ == "__main__":
    main()
