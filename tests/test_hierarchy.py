import random
from collections import defaultdict
from typing import Any, Callable

import polars as pl
import pytest
from polars_grouper import hierarchy_levels, hierarchy_paths, hierarchy_totals

HierarchyFunction = Callable[..., pl.Expr]
Edge = tuple[Any, Any, float]

CAR_BOM = pl.DataFrame(
    {
        "parent": ["car", "car", "car", "wheel", "wheel", "wheel", "rim"],
        "child": ["wheel", "steering_wheel", "screw", "tyre", "rim", "screw", "iron"],
        "qty": [4.0, 1.0, 20.0, 1.0, 1.0, 5.0, 2.5],
    }
)

PAIR_COLUMNS = ["ancestor", "descendant", "level", "quantity", "is_leaf"]
PATH_COLUMNS = ["ancestor", "descendant", "level", "parent", "quantity_per", "quantity", "is_leaf", "path"]


def _explode(df: pl.DataFrame | pl.LazyFrame, function: HierarchyFunction, **kwargs: Any) -> pl.DataFrame:
    """Run a hierarchy function over the parent/child/qty columns and return the unnested rows."""
    quantity = kwargs.pop("quantity", "qty")
    result = df.select(function("parent", "child", quantity, **kwargs).alias("bom")).unnest("bom")
    return result.collect() if isinstance(result, pl.LazyFrame) else result


def _rows(df: pl.DataFrame) -> list[tuple[Any, ...]]:
    """
    Return the rows of a frame as a sorted list of tuples.

    A list rather than a dict keeps duplicated rows visible; collapsing results into a dict is what
    let a 2x duplication bug in calculate_shortest_path go unnoticed.
    """
    return sorted(df.rows(), key=repr)


def _reference_paths(
    edges: list[Edge], top_level_only: bool = False, include_self: bool = False, max_depth: int | None = None
) -> list[tuple[Any, ...]]:
    """Enumerate every path the slow, obvious way: recursion from each root, as a WITH RECURSIVE would."""
    children: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
    nodes: list[Any] = []
    for parent, child, qty in edges:
        children[parent].append((child, qty))
        nodes.extend(n for n in (parent, child) if n not in nodes)
    has_parent = {child for _, child, _ in edges}

    rows: list[tuple[Any, ...]] = []

    def descend(root: Any, node: Any, path: list[Any], quantity: float) -> None:
        if max_depth is not None and len(path) > max_depth:
            return
        for child, per in children[node]:
            child_path = [*path, child]
            is_leaf = not children[child]
            rows.append((root, child, len(path), node, per, quantity * per, is_leaf, child_path))
            descend(root, child, child_path, quantity * per)

    for root in nodes:
        if top_level_only and root in has_parent:
            continue
        if not include_self and not children[root]:
            continue
        if include_self:
            rows.append((root, root, 0, None, None, 1.0, not children[root], [root]))
        descend(root, root, [root], 1.0)
    return rows


def _reference_levels(paths: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Aggregate reference paths to one row per (ancestor, descendant, level)."""
    quantities: dict[tuple[Any, Any, int], float] = defaultdict(float)
    leaves = {}
    for ancestor, descendant, level, _, _, quantity, is_leaf, _ in paths:
        quantities[(ancestor, descendant, level)] += quantity
        leaves[descendant] = is_leaf
    return [(a, d, level, q, leaves[d]) for (a, d, level), q in quantities.items()]


def _reference_totals(paths: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Aggregate reference paths to one row per (ancestor, descendant)."""
    quantities: dict[tuple[Any, Any], float] = defaultdict(float)
    shallowest: dict[tuple[Any, Any], int] = {}
    leaves = {}
    for ancestor, descendant, level, _, _, quantity, is_leaf, _ in paths:
        quantities[(ancestor, descendant)] += quantity
        shallowest[(ancestor, descendant)] = min(level, shallowest.get((ancestor, descendant), level))
        leaves[descendant] = is_leaf
    return [(a, d, shallowest[(a, d)], q, leaves[d]) for (a, d), q in quantities.items()]


def _random_dag(seed: int, num_nodes: int = 25, num_edges: int = 60) -> list[Edge]:
    """
    Build a random DAG with shared subassemblies and a few duplicate edges.

    Edges only point from lower to higher node numbers, which rules out cycles. Integer quantities
    keep every product and sum exact, so results can be compared without a tolerance.
    """
    rng = random.Random(seed)
    edges: list[Edge] = []
    for _ in range(num_edges):
        parent, child = sorted(rng.sample(range(num_nodes), 2))
        edges.append((f"n{parent:02d}", f"n{child:02d}", float(rng.randint(1, 4))))
    edges.extend(rng.sample(edges, 3))
    return edges


def _edges_frame(edges: list[Edge]) -> pl.DataFrame:
    return pl.DataFrame(edges, schema=["parent", "child", "qty"], orient="row")


def test_car_bom_totals() -> None:
    """Test the rolled-up totals: screws come in directly and via the wheels, iron three levels deep."""
    result = _explode(CAR_BOM, hierarchy_totals)

    assert result.columns == PAIR_COLUMNS
    assert _rows(result) == sorted(
        [
            ("car", "wheel", 1, 4.0, False),
            ("car", "steering_wheel", 1, 1.0, True),
            ("car", "screw", 1, 40.0, True),
            ("car", "tyre", 2, 4.0, True),
            ("car", "rim", 2, 4.0, False),
            ("car", "iron", 3, 10.0, True),
            ("wheel", "tyre", 1, 1.0, True),
            ("wheel", "rim", 1, 1.0, False),
            ("wheel", "screw", 1, 5.0, True),
            ("wheel", "iron", 2, 2.5, True),
            ("rim", "iron", 1, 2.5, True),
        ],
        key=repr,
    )


def test_car_bom_levels() -> None:
    """Test that screws reached at two depths below the car are kept apart per level."""
    result = _explode(CAR_BOM, hierarchy_levels, top_level_only=True)

    assert result.columns == PAIR_COLUMNS
    assert _rows(result) == sorted(
        [
            ("car", "wheel", 1, 4.0, False),
            ("car", "steering_wheel", 1, 1.0, True),
            ("car", "screw", 1, 20.0, True),
            ("car", "tyre", 2, 4.0, True),
            ("car", "rim", 2, 4.0, False),
            ("car", "screw", 2, 20.0, True),
            ("car", "iron", 3, 10.0, True),
        ],
        key=repr,
    )


def test_car_bom_paths_in_indented_order() -> None:
    """Test the path rows exactly, including their depth-first (indented BOM) order."""
    result = _explode(CAR_BOM.lazy(), hierarchy_paths, top_level_only=True)
    assert result.columns == PATH_COLUMNS
    assert result.rows() == [
        ("car", "wheel", 1, "car", 4.0, 4.0, False, ["car", "wheel"]),
        ("car", "tyre", 2, "wheel", 1.0, 4.0, True, ["car", "wheel", "tyre"]),
        ("car", "rim", 2, "wheel", 1.0, 4.0, False, ["car", "wheel", "rim"]),
        ("car", "iron", 3, "rim", 2.5, 10.0, True, ["car", "wheel", "rim", "iron"]),
        ("car", "screw", 2, "wheel", 5.0, 20.0, True, ["car", "wheel", "screw"]),
        ("car", "steering_wheel", 1, "car", 1.0, 1.0, True, ["car", "steering_wheel"]),
        ("car", "screw", 1, "car", 20.0, 20.0, True, ["car", "screw"]),
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"top_level_only": True},
        {"include_self": True},
        {"max_depth": 2},
        {"top_level_only": True, "include_self": True, "max_depth": 3},
    ],
)
def test_matches_recursive_reference(seed: int, kwargs: dict[str, Any]) -> None:
    """Test all three functions against a naive recursive explosion of a random DAG."""
    edges = _random_dag(seed)
    df = _edges_frame(edges)
    reference = _reference_paths(edges, **kwargs)

    assert _rows(_explode(df, hierarchy_paths, **kwargs)) == sorted(reference, key=repr)
    assert _rows(_explode(df, hierarchy_levels, **kwargs)) == sorted(_reference_levels(reference), key=repr)
    assert _rows(_explode(df, hierarchy_totals, **kwargs)) == sorted(_reference_totals(reference), key=repr)


def test_paths_aggregate_to_levels_and_totals() -> None:
    """Test the documented group_by recipes that turn paths into levels and levels into totals."""
    df = _edges_frame(_random_dag(seed=7))
    paths = _explode(df, hierarchy_paths)
    levels = _explode(df, hierarchy_levels)
    totals = _explode(df, hierarchy_totals)

    keys = ["ancestor", "descendant", "level"]
    assert _rows(paths.group_by(keys).agg(pl.col("quantity").sum()).select(*keys, "quantity")) == _rows(
        levels.select(*keys, "quantity")
    )
    pair = ["ancestor", "descendant"]
    rolled_up = levels.group_by(pair).agg(pl.col("level").min(), pl.col("quantity").sum())
    assert _rows(rolled_up.select(*pair, "level", "quantity")) == _rows(totals.select(*pair, "level", "quantity"))


def test_tree_has_one_path_per_pair() -> None:
    """Test that in a tree (a chart of accounts, say) every pair has exactly one path."""
    df = pl.DataFrame(
        {
            "parent": ["total", "total", "assets", "assets", "liabilities", "cash"],
            "child": ["assets", "liabilities", "cash", "receivables", "payables", "bank"],
        }
    )
    paths = _explode(df, hierarchy_paths, quantity=None)
    totals = _explode(df, hierarchy_totals, quantity=None)

    assert paths.height == totals.height == 11
    assert paths.select("ancestor", "descendant").is_unique().all()


def test_without_quantity_counts_paths() -> None:
    """Test that omitting quantities counts the distinct paths between two nodes."""
    df = pl.DataFrame({"parent": ["a", "a", "b", "c"], "child": ["b", "c", "d", "d"]})

    result = _explode(df, hierarchy_totals, quantity=None, top_level_only=True)

    assert _rows(result) == [
        ("a", "b", 1, 1.0, False),
        ("a", "c", 1, 1.0, False),
        ("a", "d", 2, 2.0, True),
    ]


def test_scalar_quantity_is_broadcast() -> None:
    """Test that a length-1 quantity applies to every edge."""
    df = pl.DataFrame({"parent": ["a", "b"], "child": ["b", "c"]})

    result = _explode(df, hierarchy_totals, quantity=pl.lit(2.0))

    assert _rows(result) == [("a", "b", 1, 2.0, False), ("a", "c", 2, 4.0, True), ("b", "c", 1, 2.0, True)]


def test_mismatched_quantity_length_raises() -> None:
    """Test that a quantity of the wrong length raises instead of truncating the edge list."""
    df = pl.DataFrame({"parent": ["a", "a", "b"], "child": ["b", "c", "d"]})

    with pytest.raises(pl.exceptions.ComputeError, match="expected 3"):
        _explode(df, hierarchy_totals, quantity=pl.Series("q", [1.0, 2.0]))


def test_duplicate_edges() -> None:
    """Test that two BOM lines for the same component add up, and stay separate paths."""
    df = pl.DataFrame({"parent": ["car", "car"], "child": ["screw", "screw"], "qty": [20.0, 4.0]})

    assert _rows(_explode(df, hierarchy_totals)) == [("car", "screw", 1, 24.0, True)]
    assert _explode(df, hierarchy_paths).get_column("quantity").to_list() == [20.0, 4.0]


def test_include_self_rows() -> None:
    """Test that include_self adds a level-0 row per node, with a null parent on paths."""
    df = pl.DataFrame({"parent": ["a"], "child": ["b"], "qty": [3.0]})

    assert _explode(df, hierarchy_paths, include_self=True).rows() == [
        ("a", "a", 0, None, None, 1.0, False, ["a"]),
        ("a", "b", 1, "a", 3.0, 3.0, True, ["a", "b"]),
        ("b", "b", 0, None, None, 1.0, True, ["b"]),
    ]
    assert _rows(_explode(df, hierarchy_totals, include_self=True)) == [
        ("a", "a", 0, 1.0, False),
        ("a", "b", 1, 3.0, True),
        ("b", "b", 0, 1.0, True),
    ]


@pytest.mark.parametrize("function", [hierarchy_totals, hierarchy_levels, hierarchy_paths])
def test_max_depth(function: HierarchyFunction) -> None:
    """Test that max_depth stops the explosion, and that 0 leaves only the self rows."""
    assert _explode(CAR_BOM, function, max_depth=1).get_column("level").max() == 1
    assert _explode(CAR_BOM, function, max_depth=0).height == 0
    assert _explode(CAR_BOM, function, max_depth=0, include_self=True).get_column("level").to_list() == [0] * 7


def test_negative_max_depth_raises() -> None:
    """Test that a negative max_depth is rejected up front."""
    with pytest.raises(ValueError, match="max_depth"):
        hierarchy_totals("parent", "child", max_depth=-1)


@pytest.mark.parametrize("function", [hierarchy_totals, hierarchy_levels, hierarchy_paths])
def test_cycle_raises(function: HierarchyFunction) -> None:
    """Test that a cycle raises an error that names it, since its quantities would be infinite."""
    df = pl.DataFrame({"parent": ["a", "b", "c", "c"], "child": ["b", "c", "a", "d"], "qty": 1.0})

    with pytest.raises(pl.exceptions.ComputeError, match="cycle.*a -> b -> c -> a"):
        _explode(df, function)


def test_self_loop_raises() -> None:
    """Test that an item listed as its own component is reported as a cycle."""
    df = pl.DataFrame({"parent": ["a", "b"], "child": ["b", "b"], "qty": 1.0})

    with pytest.raises(pl.exceptions.ComputeError, match="cycle.*b -> b"):
        _explode(df, hierarchy_totals)


@pytest.mark.parametrize(
    ("function", "columns"),
    [(hierarchy_totals, PAIR_COLUMNS), (hierarchy_levels, PAIR_COLUMNS), (hierarchy_paths, PATH_COLUMNS)],
)
def test_empty_input(function: HierarchyFunction, columns: list[str]) -> None:
    """Test that an empty edge list gives an empty frame with the full schema."""
    df = pl.DataFrame(schema={"parent": pl.String, "child": pl.String, "qty": pl.Float64})

    result = _explode(df, function)

    assert result.height == 0
    assert result.columns == columns


def test_null_parent_or_child_is_skipped() -> None:
    """Test that rows with a missing endpoint are ignored, as in the other graph functions."""
    df = pl.DataFrame({"parent": ["a", None, "b"], "child": ["b", "x", None], "qty": [2.0, 1.0, 1.0]})

    assert _rows(_explode(df, hierarchy_totals)) == [("a", "b", 1, 2.0, True)]


def test_null_quantity_raises() -> None:
    """Test that a missing quantity raises rather than silently dropping the subtree below it."""
    df = pl.DataFrame({"parent": ["a", "b"], "child": ["b", "c"], "qty": [2.0, None]})

    with pytest.raises(pl.exceptions.ComputeError, match="null for the edge b -> c"):
        _explode(df, hierarchy_totals)


def test_integer_ids_keep_their_dtype() -> None:
    """Test that integer material numbers come back as integers, including inside the path."""
    df = pl.DataFrame({"parent": [100, 100, 200], "child": [200, 300, 300], "qty": [2, 1, 3]})

    paths = _explode(df, hierarchy_paths)

    assert paths.schema["ancestor"] == pl.Int64
    assert paths.schema["parent"] == pl.Int64
    assert paths.schema["path"] == pl.List(pl.Int64)
    assert _rows(_explode(df, hierarchy_totals)) == [
        (100, 200, 1, 2.0, False),
        (100, 300, 1, 7.0, True),
        (200, 300, 1, 3.0, True),
    ]


def test_mismatched_id_dtypes_fall_back_to_string() -> None:
    """Test that parent and child columns of different dtypes produce string node columns."""
    df = pl.DataFrame({"parent": [1, 2], "child": ["2", "3"]})

    result = _explode(df, hierarchy_totals, quantity=None)

    assert result.schema["ancestor"] == pl.String
    assert _rows(result) == [("1", "2", 1, 1.0, False), ("1", "3", 2, 1.0, True), ("2", "3", 1, 1.0, True)]


@pytest.mark.parametrize("function", [hierarchy_totals, hierarchy_levels, hierarchy_paths])
def test_lazy_matches_eager(function: HierarchyFunction) -> None:
    """Test that the explosion works inside a lazy query and matches the eager result."""
    lazy = CAR_BOM.lazy().select(function("parent", "child", "qty").alias("bom")).unnest("bom")

    assert lazy.collect_schema().names() == _explode(CAR_BOM, function).columns
    assert lazy.collect().equals(_explode(CAR_BOM, function))
