from datetime import date

import numpy as np
import pandas as pd
import pytest

from batt.data.store import PointInTimeTable
from batt.data.validate import validate_prices
from batt.sim.grid import SimulationGrid
from batt.timeutils import hours, minutes

TZ = "America/Chicago"


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(TZ).tz_convert("UTC")


def test_rt_price_visible_only_after_interval_end_plus_delay(market, timing):
    start = local("2024-03-05 18:00")
    end = start + minutes(15)
    published = end + minutes(timing.rt_publish_delay_minutes)
    before = market.view(published - minutes(1 / 60)).series("rt", start, end)
    after = market.view(published).series("rt", start, end)
    assert before.empty
    assert len(after) == 1


def test_da_prices_publish_at_1330_the_day_before(market):
    day_start, day_end = local("2024-03-06 00:00"), local("2024-03-07 00:00")
    published = local("2024-03-05 13:30")
    assert market.view(published - minutes(1)).series("da", day_start, day_end).empty
    assert len(market.view(published).series("da", day_start, day_end)) == 24


def test_views_can_narrow_but_not_widen(market):
    t = local("2024-03-05 12:00")
    view = market.view(t)
    assert view.at(t - hours(1)).now < t
    with pytest.raises(ValueError):
        view.at(t + minutes(1))


def test_latest_published_vintage_wins():
    start = local("2024-03-05 18:00")
    frame = pd.DataFrame(
        {
            "interval_start_utc": [start, start],
            "value": [10.0, 99.0],
            "publish_time_utc": [start - hours(6), start - hours(1)],
        }
    )
    table = PointInTimeTable(frame, resolution_minutes=60)
    assert table.series(start - hours(3)).iloc[0] == 10.0
    assert table.series(start).iloc[0] == 99.0


def test_window_queries_match_the_general_path(market):
    """Binary-search windows (prices: one row per interval, published in order) must return
    the same rows as the general masked path used for tables with vintages."""
    rng = np.random.default_rng(5)
    for table in market.tables.values():
        general = PointInTimeTable(table.frame, table.resolution_minutes)
        general._ordered = False  # force the masked path
        assert table._ordered
        lo, hi = table.frame["interval_start_utc"].iloc[[0, -1]]
        span = (hi - lo).value
        for _ in range(30):
            now, a, b = (lo + pd.Timedelta(int(x), unit="ns") for x in rng.uniform(0, span, 3))
            for args in ((now, a, b), (now, a, None), (now, None, b), (None, a, b)):
                for fast, slow in zip(table.rows(*args), general.rows(*args), strict=True):
                    np.testing.assert_array_equal(fast, slow)


def test_dst_days_have_92_and_100_intervals():
    def n(d0, d1):
        return len(SimulationGrid.for_days(d0, d1, TZ))

    assert n(date(2024, 3, 10), date(2024, 3, 11)) == 92
    assert n(date(2024, 11, 3), date(2024, 11, 4)) == 100
    assert n(date(2024, 6, 1), date(2024, 6, 2)) == 96


@pytest.mark.parametrize("window", ["day", "week", "month"])
def test_windows_tile_the_grid(window):
    grid = SimulationGrid.for_days(date(2024, 1, 20), date(2024, 3, 15), TZ)
    slices = grid.windows(window)
    assert slices[0].start == 0
    assert slices[-1].stop == len(grid)
    assert all(a.stop == b.start for a, b in zip(slices, slices[1:], strict=False))


def test_clean_synthetic_prices_pass_validation(frames):
    rt, da = frames
    assert all(r.ok for r in validate_prices(rt, "rt"))
    assert all(r.ok for r in validate_prices(da, "da"))


def test_validation_flags_gaps_duplicates_and_bad_prices(frames):
    rt = frames[0].copy()
    rt = rt.drop(index=[100, 101, 102])
    dup = rt.iloc[[500]].copy()
    dup["price"] += 1.0
    rt = pd.concat([rt, dup], ignore_index=True)
    rt.loc[1000, "price"] = np.nan
    report = validate_prices(rt, "rt")[0]
    assert report.missing_intervals == 3
    assert report.duplicate_rows == 1
    assert report.conflicting_intervals == 1
    assert report.bad_prices == 1
    assert not report.ok
