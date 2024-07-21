from typing import List

from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql.types import LongType

from caad.spark_utils import TemporalWindow as tw


def lagged_delta(
    minuend_col: str | Column,
    subtrahend_col: str | Column,
    timestamp_col: str | Column,
    partition_col: str | List[str] | Column,
    window_length: int,
    offset: int,
    unit_of_time: str,
    subtrahend_aggregation: str,
) -> Column:

    if isinstance(minuend_col, str):
        minuend_col = f.col(minuend_col)
    if isinstance(subtrahend_col, str):
        subtrahend_col = f.col(subtrahend_col)
    if isinstance(timestamp_col, str):
        timestamp_col = f.col(timestamp_col)

    def aggregation_strategies(col):
        return {
            "mean": f.mean(col),
            "last": f.last(col, ignorenulls=True),
            "first": f.first(col, ignorenulls=True),
            "median": f.percentile_approx(col, 0.5),
        }

    valid_agg_strategies = aggregation_strategies(subtrahend_col).keys()

    subtrahend_col = aggregation_strategies(subtrahend_col).get(
        subtrahend_aggregation, None
    )

    if subtrahend_col is None:
        raise ValueError(
            f"Invalid aggregation strategy; may only be one of: {valid_agg_strategies}."
        )

    w = tw.create_lookback_window(
        window_length=window_length,
        offset=offset,
        unit_of_time=unit_of_time,
        timestamp_col=timestamp_col,
        partition_col=partition_col,
    )

    return minuend_col - subtrahend_col.over(w)


def rate_of_change(
    col_name: str | Column,
    timestamp_col: str | Column,
    partition_col: str | List[str] | Column,
    window_length: int,
    offset: int,
    unit_of_time: str,
    subtrahend_aggregation: str,
) -> Column:

    if isinstance(timestamp_col, str):
        timestamp_col = f.col(timestamp_col)

    numerator = lagged_delta(
        minuend_col=col_name,
        subtrahend_col=col_name,
        timestamp_col=timestamp_col,
        partition_col=partition_col,
        window_length=window_length,
        offset=offset,
        unit_of_time=unit_of_time,
        subtrahend_aggregation=subtrahend_aggregation,
    )

    denominator = lagged_delta(
        minuend_col=timestamp_col.cast(LongType),
        subtrahend_col=timestamp_col.cast(LongType),
        timestamp_col=timestamp_col,
        partition_col=partition_col,
        window_length=window_length,
        offset=offset,
        unit_of_time=unit_of_time,
        subtrahend_aggregation="latest",
    )

    return numerator / denominator
