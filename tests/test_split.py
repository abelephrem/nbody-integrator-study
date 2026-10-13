from collections import Counter

import pytest

from gnn.split import SEED, load_split, make_split, read_tags, setting_key


@pytest.fixture(scope="module")
def tags():
    """{filename: tag dictionary} for all 360 files, read once."""
    return {t["file"]: t for t in read_tags()}


@pytest.fixture(scope="module")
def split():
    """The saved split from gnn/splits/split.json - the one every later stage uses."""
    return load_split()


def test_split_totals(split):
    assert len(split["train"]) == 208
    assert len(split["val"]) == 36
    assert len(split["test"]) == 36
    assert len(split["holdout"]) == 80


def test_no_file_in_two_sets(split):
    sets = [set(split[k]) for k in ("train", "val", "test", "holdout")]
    for i in range(4):
        for j in range(i + 1, 4):                  # every pair of sets, each pair once
            assert not (sets[i] & sets[j])         # & = files in both; must be empty
    assert len(set().union(*sets)) == 360          # every file accounted for, none missing


def test_holdout_kept_separate(split, tags):
    for k in ("train", "val", "test"):
        assert all(tags[f]["split"] == "train" for f in split[k]), k   # only train-tagged files
    assert all(tags[f]["split"] != "train" for f in split["holdout"])  # every other tag is holdout


def test_every_setting_in_each_set(split, tags):
    all_settings = {setting_key(t) for t in tags.values() if t["split"] == "train"}
    assert len(all_settings) == 36                 # 16 two-body + 20 cluster combinations

    train = Counter(setting_key(tags[f]) for f in split["train"])
    assert set(train) == all_settings              # every setting appears in train

    for k in ("val", "test"):
        counts = Counter(setting_key(tags[f]) for f in split[k])
        assert set(counts) == all_settings, k      # every setting appears...
        assert set(counts.values()) == {1}, k      # ...exactly once


def test_split_is_reproducible(split, tags):
    # re-run the split from scratch with the saved seed: must match the saved file exactly
    remade = make_split(list(tags.values()), split["seed"])
    for k in ("train", "val", "test", "holdout"):
        assert remade[k] == split[k], k
    assert split["seed"] == SEED
