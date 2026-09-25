"""
End-to-end use of the hierarchy explosions on a small bike factory's ERP data.

Each test answers one business question with a complete lazy pipeline: scan the bill of materials from
Parquet, explode it, join it to the other tables and collect once at the end.

The BOM has the shapes that make explosion hard: screws are used at several levels, the wheel is a
subassembly shared by both products, and aluminium sits three levels below them.
"""

from pathlib import Path
from typing import Literal

import polars as pl
import pytest
from polars_grouper import hierarchy_levels, hierarchy_paths, hierarchy_totals

# (assembly, component, quantity per assembly)
BOM_LINES = [
    ("bike", "frame", 1.0),
    ("bike", "wheel", 2.0),
    ("bike", "screw", 10.0),
    ("bike", "handlebar", 1.0),
    ("ebike", "frame", 1.0),
    ("ebike", "wheel", 2.0),
    ("ebike", "battery", 1.0),
    ("ebike", "motor", 1.0),
    ("ebike", "screw", 12.0),
    ("frame", "steel_tube", 3.5),
    ("frame", "screw", 6.0),
    ("wheel", "rim", 1.0),
    ("wheel", "spoke", 32.0),
    ("wheel", "tyre", 1.0),
    ("wheel", "screw", 2.0),
    ("rim", "aluminium", 0.75),
    ("handlebar", "steel_tube", 0.5),
    ("handlebar", "grip", 2.0),
    ("motor", "copper_wire", 12.0),
    ("motor", "screw", 4.0),
]

PRODUCTION_PLAN = pl.LazyFrame({"product": ["bike", "ebike"], "units": [100, 50]})


@pytest.fixture
def bom(tmp_path: Path) -> pl.LazyFrame:
    """Write the bill of materials as an ERP export, including columns the explosion does not need."""
    path = tmp_path / "bom.parquet"
    (
        pl.DataFrame(BOM_LINES, schema=["assembly", "component", "qty"], orient="row")
        .with_columns(plant=pl.lit("NL01"), bom_item=pl.int_range(pl.len()) * 10 + 10)
        .write_parquet(path)
    )
    return pl.scan_parquet(path)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_purchase_requirements_for_a_production_plan(
    bom: pl.LazyFrame, engine: Literal["in-memory", "streaming"]
) -> None:
    """
    Compute what to buy to build 100 bikes and 50 e-bikes.

    Explode each product down to its purchased parts (the leaves), multiply by the planned units, add up
    per part across products and subtract what is already in stock.
    """
    stock = pl.LazyFrame({"part": ["screw", "tyre", "battery"], "on_hand": [1000, 150, 10]})

    to_buy = (
        bom.select(hierarchy_totals("assembly", "component", "qty", top_level_only=True).alias("bom"))
        .unnest("bom")
        .filter(pl.col("is_leaf"))
        .join(PRODUCTION_PLAN, left_on="ancestor", right_on="product")
        .group_by(part="descendant")
        .agg(needed=(pl.col("quantity") * pl.col("units")).sum())
        .join(stock, on="part", how="left")
        .select("part", to_buy=pl.col("needed") - pl.col("on_hand").fill_null(0))
        .collect(engine=engine)
    )

    assert dict(to_buy.iter_rows()) == {
        "screw": 100 * 20 + 50 * 26 - 1000,
        "spoke": 150 * 64,
        "tyre": 150 * 2 - 150,
        "steel_tube": 100 * 4.0 + 50 * 3.5,
        "aluminium": 150 * 1.5,
        "grip": 100 * 2,
        "battery": 50 - 10,
        "copper_wire": 50 * 12,
    }


def test_cost_roll_up(bom: pl.LazyFrame) -> None:
    """
    Roll purchase prices up into the cost of every assembly.

    Join each assembly's leaf components to their purchase prices and sum quantity x price. Every
    subassembly is costed too, because by default every node with components is exploded.
    """
    prices = pl.LazyFrame(
        {
            "part": ["screw", "spoke", "tyre", "steel_tube", "aluminium", "grip", "battery", "copper_wire"],
            "price": [0.10, 0.25, 15.0, 8.0, 4.0, 2.5, 300.0, 0.5],
        }
    )

    cost = (
        bom.select(hierarchy_totals("assembly", "component", "qty").alias("bom"))
        .unnest("bom")
        .filter(pl.col("is_leaf"))
        .join(prices, left_on="descendant", right_on="part")
        .group_by(assembly="ancestor")
        .agg(cost=(pl.col("quantity") * pl.col("price")).sum())
        .collect()
    )

    assert dict(cost.iter_rows()) == pytest.approx(
        {
            "rim": 3.0,
            "wheel": 3.0 + 8.0 + 15.0 + 0.2,
            "frame": 28.0 + 0.6,
            "handlebar": 4.0 + 5.0,
            "motor": 6.0 + 0.4,
            "bike": 28.6 + 2 * 26.2 + 1.0 + 9.0,
            "ebike": 28.6 + 2 * 26.2 + 300.0 + 6.4 + 1.2,
        }
    )


def test_where_used(bom: pl.LazyFrame) -> None:
    """
    Find the assemblies affected by a late spoke delivery, and how many spokes each needs.

    The same explosion answers the reverse question by filtering on the component instead of the product.
    """
    affected = (
        bom.select(hierarchy_totals("assembly", "component", "qty").alias("bom"))
        .unnest("bom")
        .filter(pl.col("descendant") == "spoke")
        .select("ancestor", "quantity")
        .collect()
    )

    assert sorted(affected.rows()) == [("bike", 64.0), ("ebike", 64.0), ("wheel", 32.0)]


def test_low_level_codes_for_mrp(bom: pl.LazyFrame) -> None:
    """
    Derive the low-level codes that set the order in which MRP plans items.

    MRP plans an item only after every assembly that uses it, so it sorts items by their low-level code:
    the deepest level at which they occur in any product. Screws are fitted directly to the bike
    (level 1) but also inside the wheel (level 2), so they are planned at level 2. A single total per
    pair can't tell you this; the per-level rows can.
    """
    codes = (
        bom.select(
            hierarchy_levels("assembly", "component", "qty", top_level_only=True, include_self=True).alias("bom")
        )
        .unnest("bom")
        .group_by(item="descendant")
        .agg(low_level_code=pl.col("level").max())
        .collect()
    )

    assert dict(codes.iter_rows()) == {
        "bike": 0,
        "ebike": 0,
        "frame": 1,
        "wheel": 1,
        "handlebar": 1,
        "battery": 1,
        "motor": 1,
        "rim": 2,
        "spoke": 2,
        "tyre": 2,
        "steel_tube": 2,
        "grip": 2,
        "copper_wire": 2,
        "screw": 2,
        "aluminium": 3,
    }


def test_screws_per_assembly_stage(bom: pl.LazyFrame) -> None:
    """
    Split the screws to stage between final assembly and sub-assembly.

    The stages run at different times, so the production plan's screws are split by level instead of
    being summed into one total.
    """
    per_stage = (
        bom.select(hierarchy_levels("assembly", "component", "qty", top_level_only=True).alias("bom"))
        .unnest("bom")
        .filter(pl.col("descendant") == "screw")
        .join(PRODUCTION_PLAN, left_on="ancestor", right_on="product")
        .group_by("level")
        .agg(screws=(pl.col("quantity") * pl.col("units")).sum())
        .collect()
    )

    assert dict(per_stage.iter_rows()) == {
        1: 100 * 10 + 50 * 12,
        2: 100 * (6 + 2 * 2) + 50 * (6 + 2 * 2 + 4),
    }


def test_indented_bom_report(bom: pl.LazyFrame) -> None:
    """Print a product's multi-level BOM the way an ERP shows it: depth-first, indented by level."""
    lines = (
        bom.select(hierarchy_paths("assembly", "component", "qty", top_level_only=True).alias("bom"))
        .unnest("bom")
        .filter(pl.col("ancestor") == "bike")
        .select("level", "descendant", "quantity_per")
        .collect()
    )

    report = ["  " * (level - 1) + f"{part} x{qty:g}" for level, part, qty in lines.iter_rows()]

    assert report == [
        "frame x1",
        "  steel_tube x3.5",
        "  screw x6",
        "wheel x2",
        "  rim x1",
        "    aluminium x0.75",
        "  spoke x32",
        "  tyre x1",
        "  screw x2",
        "screw x10",
        "handlebar x1",
        "  steel_tube x0.5",
        "  grip x2",
    ]


def test_trace_where_a_quantity_comes_from(bom: pl.LazyFrame) -> None:
    """
    Trace why a bike needs 20 screws.

    Every route from the bike to a screw, with the quantity it contributes; together they make up the
    total that hierarchy_totals reports.
    """
    exploded = bom.select(hierarchy_paths("assembly", "component", "qty").alias("bom")).unnest("bom")
    routes = (
        exploded.filter(pl.col("ancestor") == "bike", pl.col("descendant") == "screw")
        .select(route=pl.col("path").list.join(" > "), quantity="quantity")
        .collect()
    )
    total = (
        bom.select(hierarchy_totals("assembly", "component", "qty").alias("bom"))
        .unnest("bom")
        .filter(pl.col("ancestor") == "bike", pl.col("descendant") == "screw")
        .collect()
    )

    assert routes.rows() == [
        ("bike > frame > screw", 6.0),
        ("bike > wheel > screw", 4.0),
        ("bike > screw", 10.0),
    ]
    assert routes.get_column("quantity").sum() == total.item(0, "quantity") == 20.0


def test_general_ledger_roll_up() -> None:
    """
    Roll general-ledger balances up the chart of accounts.

    Link every account to itself and to all accounts below it (include_self), join the journal on the
    lowest level and sum per higher-level account. There are no quantities, so none is passed. Account
    numbers stay integers, so the join to the journal needs no casting.
    """
    breakpoint()
    chart_of_accounts = pl.LazyFrame(
        {"parent_account": [1000, 1000, 1100, 1100, 1200], "account": [1100, 1200, 1110, 1120, 1210]}
    )
    journal = pl.LazyFrame({"account": [1110, 1110, 1120, 1210], "amount": [100, 50, 30, 500]})

    balances = (
        chart_of_accounts.select(hierarchy_totals("parent_account", "account", include_self=True).alias("closure"))
        .unnest("closure")
        .join(journal, left_on="descendant", right_on="account")
        .group_by(account="ancestor")
        .agg(pl.col("amount").sum())
        .collect()
    )

    assert dict(balances.iter_rows()) == {1000: 680, 1100: 180, 1110: 150, 1120: 30, 1200: 500, 1210: 500}


def test_pipeline_is_lazy_until_collect(tmp_path: Path) -> None:
    """
    Check that nothing is exploded before collect, while the schema is already known.

    The BOM contains a cycle, which can only be detected by running the explosion; building the query
    and resolving its schema must succeed, and collecting it must fail.
    """
    path = tmp_path / "cyclic_bom.parquet"
    pl.DataFrame({"assembly": ["a", "b"], "component": ["b", "a"], "qty": [1.0, 1.0]}).write_parquet(path)

    requirements = (
        pl.scan_parquet(path)
        .select(hierarchy_totals("assembly", "component", "qty").alias("bom"))
        .unnest("bom")
        .filter(pl.col("is_leaf"))
        .group_by("descendant")
        .agg(pl.col("quantity").sum())
    )

    assert requirements.collect_schema() == pl.Schema({"descendant": pl.String, "quantity": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match="cycle"):
        requirements.collect()
