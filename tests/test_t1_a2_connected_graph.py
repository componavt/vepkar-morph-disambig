"""Focused checks for the A2 connected-graph draft."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "t1_a2_connected_graph", ROOT / "scripts/t1-a2-connected-graph.py"
)
assert SPEC is not None and SPEC.loader is not None
a2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a2)


def test_shared_log_area_scale_uses_full_three_to_one_area_range():
    minimum = 29
    maximum = 932
    low = a2.node_area(minimum, minimum, maximum)
    high = a2.node_area(maximum, minimum, maximum)
    halfway = a2.node_area(
        round(math.sqrt((minimum + 1) * (maximum + 1)) - 1), minimum, maximum
    )
    assert math.isclose(high / low, 3.0)
    assert math.isclose(halfway, (low + high) / 2, rel_tol=0.015)
    assert a2.node_radius(932, minimum, maximum) > a2.node_radius(29, minimum, maximum)


def test_percent_labels_keep_sub_one_percent_precision():
    assert a2.format_percent(3 / 870) == "0,34%"
    assert a2.format_percent(9 / 236) == "3,8%"
