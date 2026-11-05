import pytest

from regrag.evals.split import split_items

ITEMS = [{"id": f"g{i:02d}", "tag": t} for i, t in enumerate(
    ["lookup"] * 8 + ["table"] * 4 + ["multi-hop"] * 6 + ["unanswerable"] * 10)]


def test_disjoint_and_complete():
    train, test = split_items(ITEMS)
    ids = [i["id"] for i in train + test]
    assert sorted(ids) == sorted(i["id"] for i in ITEMS) and not {i["id"] for i in train} & {i["id"] for i in test}


def test_every_tag_in_both_parts():
    train, test = split_items(ITEMS)
    tags = {i["tag"] for i in ITEMS}
    assert {i["tag"] for i in train} == tags and {i["tag"] for i in test} == tags


def test_same_seed_same_split_other_seed_differs():
    a, b = split_items(ITEMS, seed=1), split_items(ITEMS, seed=1)
    assert a == b and split_items(ITEMS, seed=2) != a


def test_original_order_kept_and_bad_fraction_rejected():
    train, _ = split_items(ITEMS)
    assert [i["id"] for i in train] == sorted(i["id"] for i in train)
    with pytest.raises(ValueError):
        split_items(ITEMS, train_fraction=1.0)
