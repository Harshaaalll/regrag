"""Sync a local folder with an S3 prefix (Floci locally, real S3 if you change the endpoint)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from regrag.config import load_config


def make_client(cfg: dict | None = None):
    import boto3

    s = (cfg or load_config())["storage"]
    return boto3.client(
        "s3",
        endpoint_url=s.get("endpoint_url") or None,  # AWS_ENDPOINT_URL env still works if this is empty
        region_name=s.get("region", "us-east-1"),
        aws_access_key_id=s.get("access_key") or None,
        aws_secret_access_key=s.get("secret_key") or None,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=bucket)


def list_remote(client, bucket: str, prefix: str) -> dict[str, dict]:
    """name (relative to prefix) -> {size, sha256 or None}."""
    out: dict[str, dict] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            name = o["Key"][len(prefix):]
            if name:
                meta = client.head_object(Bucket=bucket, Key=o["Key"]).get("Metadata", {})
                out[name] = {"size": o["Size"], "sha256": meta.get("sha256")}
    return out


def push(client, folder: Path, bucket: str, prefix: str) -> dict[str, list[str]]:
    """Upload new or changed files. Returns {'uploaded': [...], 'skipped': [...]}."""
    ensure_bucket(client, bucket)
    remote = list_remote(client, bucket, prefix)
    done: dict[str, list[str]] = {"uploaded": [], "skipped": []}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        digest = sha256(p)
        if remote.get(p.name, {}).get("sha256") == digest:
            done["skipped"].append(p.name)
            continue
        client.upload_file(str(p), bucket, prefix + p.name, ExtraArgs={"Metadata": {"sha256": digest}})
        done["uploaded"].append(p.name)
    return done


def pull(client, folder: Path, bucket: str, prefix: str) -> dict[str, list[str]]:
    """Download files that are missing locally or differ. Returns {'downloaded': [...], 'skipped': [...]}."""
    folder.mkdir(parents=True, exist_ok=True)
    done: dict[str, list[str]] = {"downloaded": [], "skipped": []}
    for name, info in sorted(list_remote(client, bucket, prefix).items()):
        dest = folder / name
        if dest.exists() and info["sha256"] and sha256(dest) == info["sha256"]:
            done["skipped"].append(name)
            continue
        client.download_file(bucket, prefix + name, str(dest))
        done["downloaded"].append(name)
    return done
