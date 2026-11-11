import numpy as np
import pytest

pytest.importorskip("faiss")

from regrag.types import Chunk
from regrag.vectordb.faiss_store import FaissStore


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return (v / np.linalg.norm(v)).tolist()


CH = [Chunk(id="a:c0", doc_id="A", text="alpha", page=1), Chunk(id="a:c1", doc_id="A", text="beta", page=2),
      Chunk(id="b:c0", doc_id="B", text="gamma", page=3)]
VEC = [_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 1, 0])]


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_search_order_filter_and_counts(tmp_path, kind):
    s = FaissStore(tmp_path, kind)
    s.add(CH, VEC)
    top = s.search(_unit([1, 0, 0]), k=2)
    assert [c.id for c, _ in top] == ["a:c0", "a:c1"] and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [c.id for c, _ in s.search(_unit([1, 0, 0]), k=5, doc_ids=["B"])] == ["b:c0"]
    assert s.count() == 3 and s.doc_counts() == {"A": 2, "B": 1}


def test_persists_and_upserts(tmp_path):
    s = FaissStore(tmp_path, "flat")
    s.add(CH, VEC)
    s.add([CH[0]], [_unit([0, 0, 1])])  # replace a:c0
    again = FaissStore(tmp_path, "flat")
    assert again.count() == 3
    assert again.search(_unit([0, 0, 1]), k=1)[0][0].id == "a:c0"


def test_ids_and_delete_doc(tmp_path):
    s = FaissStore(tmp_path, "flat")
    s.add(CH, VEC)
    assert s.ids() == {"a:c0", "a:c1", "b:c0"}
    s.delete_doc("A")
    assert s.ids() == {"b:c0"} and FaissStore(tmp_path, "flat").count() == 1
