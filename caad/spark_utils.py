from pyspark.sql import Column, Window


class TemporalWindow:
    resolution = {
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

        multiplier = TemporalWindow.resolution.get(unit_of_time, None)

        if multiplier is None:
            raise ValueError(
                f"Unit of time must be one of: {TemporalWindow.resolution.keys()}"
            )

        return multiplier * window_length

    @staticmethod
    def create_lookback_window(
        window_length: int,
        unit_of_time: str,
        order_col: list[str] | str | Column,
        partition_col: list[str] | str | Column,
        include_current: bool = True,
        offset: int = 0
    ):
        """Return a pyspark `WindowSpec` object that's configured to perform a lookback operation across time,
        inclusive of the current time.

        Args:
            window_length (int): The length of the window.
            unit_of_time (str): The unit of time for the window length.
            order_col (list[str] | str | Column): The column to order the window by.
            partition_col (list[str] | str | Column): The column to partition the window by.
            include_current (bool, optional): Boolean flag that indicates if current row is included. Defaults to True.
            offset (int, optional): The offset to apply to the bounds of the window.

        Returns:
            pyspark.sql.Window: A window to be used by pyspark functions.
        """

        lookback_window_length = TemporalWindow.convert_window_length_to_seconds(
            window_length=window_length, unit_of_time=unit_of_time
        )
        rescaled_offset = TemporalWindow.convert_window_length_to_seconds(
            window_length=offset, unit_of_time=unit_of_time
        )

        start_bound = -(lookback_window_length + rescaled_offset)

        if include_current:
            end_bound = Window.currentRow - rescaled_offset
        else:
            end_bound = Window.currentRow - 1 - rescaled_offset

        return (
            Window.partitionBy(partition_col)
            .orderBy(order_col)
            .rangeBetween(start=start_bound, end=end_bound)
        )

    @staticmethod
    def create_lookahead_window(
        window_length: int,
        unit_of_time: str,
        order_col: list[str] | str | Column,
        partition_col: list[str] | str | Column,
        include_current: bool = False,
        offset: int = 0
    ):
        """Return a pyspark `WindowSpec` object that's configured to perform a lookahead operation across time,
        exclusive of the current time.

        Args:
            window_length (int): The length of the window.
            unit_of_time (str): The unit of time for the window length.
            order_col (list[str] | str | Column): The column to order the window by.
            partition_col (list[str] | str | Column): The column to partition the window by.
            include_current (bool, optional): Boolean flag that indicates if current row is included. Defaults to True.
            offset (int, optional): The offset to apply to the bounds of the window.

        Returns:
            pyspark.sql.Window: A window to be used by pyspark functions.
        """

        lookahead_window_length = TemporalWindow.convert_window_length_to_seconds(
            window_length=window_length, unit_of_time=unit_of_time
        )

        rescaled_offset = TemporalWindow.convert_window_length_to_seconds(
            window_length=offset, unit_of_time=unit_of_time
        )

        end_bound = lookahead_window_length + rescaled_offset

        if include_current:
            start_bound = Window.currentRow + rescaled_offset
        else:
            start_bound = Window.currentRow + 1 + rescaled_offset

        return (
            Window.partitionBy(partition_col)
            .orderBy(order_col)
            .rangeBetween(start=start_bound, end=end_bound)
        )
