from __future__ import annotations

from regrag.storage import s3


class FakeS3:
    class exceptions:
        ClientError = Exception

    def __init__(self):
        self.objs: dict[str, tuple[bytes, dict]] = {}

    def head_bucket(self, Bucket): ...
    def create_bucket(self, Bucket): ...

    def get_paginator(self, _):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v[0])} for k, v in outer.objs.items() if k.startswith(Prefix)]}

        return P()

    def head_object(self, Bucket, Key):
        return {"Metadata": self.objs[Key][1]}

    def upload_file(self, path, bucket, key, ExtraArgs):
        self.objs[key] = (open(path, "rb").read(), ExtraArgs["Metadata"])

    def download_file(self, bucket, key, path):
        open(path, "wb").write(self.objs[key][0])


def test_push_is_idempotent_and_detects_changes(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"one")
    c = FakeS3()
    assert s3.push(c, tmp_path, "b", "raw/")["uploaded"] == ["a.pdf"]
    assert s3.push(c, tmp_path, "b", "raw/") == {"uploaded": [], "skipped": ["a.pdf"]}
    (tmp_path / "a.pdf").write_bytes(b"two")
    assert s3.push(c, tmp_path, "b", "raw/")["uploaded"] == ["a.pdf"]


def test_pull_restores_missing_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"data")
    c = FakeS3()
    s3.push(c, src, "b", "raw/")
    assert s3.pull(c, dst, "b", "raw/")["downloaded"] == ["a.pdf"]
    assert (dst / "a.pdf").read_bytes() == b"data"
    assert s3.pull(c, dst, "b", "raw/")["skipped"] == ["a.pdf"]
