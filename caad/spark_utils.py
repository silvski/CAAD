from typing import List

from pyspark.sql import Column, Window
from pyspark.sql import functions as f
from pyspark.sql.types import LongType


class TemporalWindow:
    mapping = {
        "days": 86400,
        "hours": 3600,
        "weeks": 604800,
        "minutes": 60,
    }

    @staticmethod
    def convert_window_length_to_seconds(window_length: int, unit_of_time: str) -> int:
        """Return the number of seconds inside a given window size."""
        if window_length < 0:
            raise ValueError("Window size must not be negative.")

        multiplier = TemporalWindow.mapping.get(unit_of_time, None)

        if multiplier is None:
            raise ValueError(
                f"Unit of time must be one of: {TemporalWindow.mapping.keys()}"
            )

        return multiplier * window_length

    @staticmethod
    def create_window(
        lookback_length: int,
        lookahead_length: int,
        lookback_offset: int,
        lookahead_offset: int,
        unit_of_time: str,
        timestamp_col: Column,
        partition_col: str | List[str] | Column,
        include_start_bound: bool = True,
        include_end_bound: bool = True,
    ):

        int_params = [
            lookback_length,
            lookahead_length,
            lookback_offset,
            lookahead_offset,
        ]
        lookback_length, lookahead_length, lookback_offset, lookahead_offset = [
            TemporalWindow.convert_window_length_to_seconds(
                window_length=param, unit_of_time=unit_of_time
            )
            for param in int_params
        ]

        if include_start_bound:
            start_bound = -(lookback_length + lookback_offset)
        else:
            start_bound = -(lookback_length + lookback_offset) + 1

        if include_end_bound:
            end_bound = lookahead_length + lookahead_offset
        else:
            end_bound = lookahead_length + lookahead_offset - 1

        return (
            Window.partitionBy(partition_col)
            .orderBy(timestamp_col.cast(LongType))
            .rangeBetween(start=start_bound, end=end_bound)
        )

    @staticmethod
    def create_lookback_window(
        window_length: int,
        offset,
        unit_of_time: str,
        timestamp_col: str | Column,
        partition_col: str | List[str] | Column,
        include_start_bound: bool = True,
        include_end_bound: bool = True,
    ):
        """Return a pyspark `WindowSpec` object that's configured to perform a lookback operation across time,
        inclusive of the current time.
        """

        return TemporalWindow.create_window(
            lookback_length=window_length,
            lookahead_length=Window.currentRow,
            lookback_offset=offset,
            lookahead_offset=0,
            unit_of_time=unit_of_time,
            timestamp_col=timestamp_col,
            partition_col=partition_col,
            include_start_bound=include_start_bound,
            include_end_bound=include_end_bound,
        )

    @staticmethod
    def create_lookahead_window(
        window_length: int,
        offset: int,
        unit_of_time: str,
        timestamp_col: str | Column,
        partition_col: str | List[str] | Column,
        include_start_bound: bool = False,
        include_end_bound: bool = True,
    ):
        """Return a pyspark `WindowSpec` object that's configured to perform a lookahead operation across time,
        exclusive of the current time.
        """
        return TemporalWindow.create_window(
            lookback_length=Window.currentRow,
            lookahead_length=window_length,
            lookback_offset=0,
            lookahead_offset=offset,
            unit_of_time=unit_of_time,
            timestamp_col=timestamp_col,
            partition_col=partition_col,
            include_start_bound=include_start_bound,
            include_end_bound=include_end_bound,
        )
