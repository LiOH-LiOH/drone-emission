import pandas as pd
import pytest

from drone_emission.analysis.weighted_quantile import weighted_csv_quantile


def test_weighted_csv_quantile_is_exact_across_chunks(tmp_path):
    path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "road_distance_m": [4000, 1000, 3000, 2000, -1, "bad"],
            "annual_sales": [1, 2, 6, 1, 100, 100],
        }
    ).to_csv(path, index=False)

    result = weighted_csv_quantile(
        path,
        value_column="road_distance_m",
        weight_column="annual_sales",
        quantile=0.75,
        chunk_rows=2,
        cache_dir=tmp_path / "cache",
    )

    assert result.value == 3000
    assert result.valid_rows == 4
    assert result.total_weight == pytest.approx(10)


def test_weighted_csv_quantile_rejects_missing_columns(tmp_path):
    path = tmp_path / "od.csv"
    pd.DataFrame({"distance": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        weighted_csv_quantile(
            path,
            value_column="road_distance_m",
            weight_column="annual_sales",
            quantile=0.75,
            cache_dir=tmp_path / "cache",
        )


def test_weighted_csv_quantile_reuses_cache(tmp_path, monkeypatch):
    path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "road_distance_m": [1000.0, 2000.0, 3000.0],
            "annual_sales": [1.0, 2.0, 7.0],
        }
    ).to_csv(path, index=False)
    kwargs = {
        "value_column": "road_distance_m",
        "weight_column": "annual_sales",
        "quantile": 0.75,
        "cache_dir": tmp_path / "cache",
    }
    first = weighted_csv_quantile(path, **kwargs)

    monkeypatch.setattr(
        "drone_emission.analysis.weighted_quantile.pd.read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CSV should not be read on a cache hit")
        ),
    )
    second = weighted_csv_quantile(path, **kwargs)
    assert second == first
