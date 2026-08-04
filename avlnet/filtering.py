"""Declarative selection of a training subset.

A run's config may carry a `filter` key naming which segments to train on:

    filter:
      type: random
      n: 9829
      seed: 42

    filter:
      type: from_list
      path: data/filelists/how2-train-alignment-scores.jsonl
      predicate:
        field: denseav_score
        op: ">"
        value: 3.0

YAML has no lambdas, so predicates are data too: a leaf is `{field, op, value}`, and
leaves combine through `{all: [...]}`, `{any: [...]}` and `{not: ...}`. Nothing is
`eval`'d.

A filter resolves to positional row indices into the grouped embedding arrays, using
the segment ids recovered by `make_grouped_ids.py`.
"""

import json
import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda a, b: a in b,
}


def evaluate_predicate(spec: dict, record: dict) -> bool:
    """Is `record` selected by `spec`? See the module docstring for the grammar."""
    if not isinstance(spec, dict):
        raise ValueError(f"Predicate must be a mapping, got {type(spec).__name__}: {spec!r}")

    if "all" in spec:
        return all(evaluate_predicate(s, record) for s in spec["all"])
    if "any" in spec:
        return any(evaluate_predicate(s, record) for s in spec["any"])
    if "not" in spec:
        return not evaluate_predicate(spec["not"], record)

    missing = {"field", "op", "value"} - set(spec)
    if missing:
        raise ValueError(
            f"Predicate {spec!r} is missing {sorted(missing)}. Expected a leaf "
            f"`{{field, op, value}}` or a compound `all`/`any`/`not`."
        )
    if spec["op"] not in OPS:
        raise ValueError(f"Unknown op `{spec['op']}` in {spec!r}. Known: {sorted(OPS)}.")
    if spec["field"] not in record:
        raise ValueError(
            f"Field `{spec['field']}` not in record with fields {sorted(record)}."
        )
    return OPS[spec["op"]](record[spec["field"]], spec["value"])


@dataclass
class Random:
    """A uniformly random subset of `n` segments, reproducible given `seed`."""

    n: int
    seed: int = 42

    def select(self, ids: list[str]) -> np.ndarray:
        if self.n > len(ids):
            raise ValueError(f"Cannot sample n={self.n} from {len(ids)} segments.")
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}.")
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(len(ids), size=self.n, replace=False))


@dataclass
class FromList:
    """The segments whose record in a JSONL scores file satisfies `predicate`.

    Records are keyed by the stem of their `video_path`. Records with no embedding,
    and embeddings with no record, are both dropped -- the selection is the
    intersection.
    """

    path: str
    predicate: dict

    def select(self, ids: list[str]) -> np.ndarray:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Scores file not found: {path}")

        position = {segment_id: i for i, segment_id in enumerate(ids)}
        indices = []
        n_records = n_passed = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                n_records += 1
                if not evaluate_predicate(self.predicate, record):
                    continue
                n_passed += 1
                i = position.get(Path(record["video_path"]).stem)
                if i is not None:
                    indices.append(i)

        print(
            f"Filter: {n_passed}/{n_records} records in {path.name} pass the predicate; "
            f"{len(indices)} of them have embeddings."
        )
        return np.sort(np.array(indices, dtype=np.int64))


FILTERS = {"random": Random, "from_list": FromList}


def parse_filter(spec: dict | None):
    """Build a filter from its config mapping; `None` means train on everything."""
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in FILTERS:
        raise ValueError(f"Unknown filter type `{kind}`. Known: {sorted(FILTERS)}.")
    try:
        return FILTERS[kind](**spec)
    except TypeError as e:
        raise ValueError(f"Bad arguments for filter type `{kind}`: {e}") from e


def select_indices(spec: dict | None, ids: list[str], min_size: int = 1) -> np.ndarray | None:
    """Resolve `spec` against `ids`, or `None` when there is no filter.

    `min_size` guards the case where a subset is smaller than a batch and
    `drop_last=True` would silently yield zero steps per epoch.
    """
    data_filter = parse_filter(spec)
    if data_filter is None:
        return None

    indices = data_filter.select(ids)
    if len(indices) == 0:
        raise ValueError(f"Filter {data_filter} selected no segments.")
    if len(indices) < min_size:
        raise ValueError(
            f"Filter {data_filter} selected {len(indices)} segments, fewer than "
            f"min_size={min_size}. With drop_last=True this trains on nothing; "
            f"lower `batch_size` or loosen the filter."
        )
    return indices
