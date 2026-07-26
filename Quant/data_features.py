"""
Market Data and Feature Engineering
===================================

This file handles two responsibilities:

1. Downloading and combining historical market data
2. Creating quantitative features used by the machine-learning models

The feature calculations use only current and previous observations.
Future information is used only to create the prediction target.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# DATA DOWNLOADING
# ============================================================


def _calculate_download_end_date(settings: Any) -> str:
    """
    Calculate the exclusive Yahoo Finance download end date.

    Historical mode downloads beyond the cutoff date so the requested
    future trading days can be evaluated.

    Live mode downloads through approximately the current date.
    """

    if settings.mode == "live":
        end_date = date.today() + timedelta(days=2)
        return end_date.strftime("%Y-%m-%d")

    cutoff = pd.Timestamp(settings.cutoff_date)

    minimum_buffer = settings.prediction_days * 4 + 20
    configured_buffer = settings.future_download_buffer_days

    buffer_days = max(minimum_buffer, configured_buffer)

    end_date = cutoff + pd.Timedelta(days=buffer_days)

    return end_date.strftime("%Y-%m-%d")


def _flatten_download_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance MultiIndex columns into ordinary column names.

    Depending on the yfinance version and number of tickers requested,
    downloaded data may have either standard columns or MultiIndex columns.
    """

    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    return result


def _remove_timezone(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-naive DatetimeIndex."""

    converted = pd.to_datetime(index)

    if converted.tz is not None:
        converted = converted.tz_localize(None)

    return converted


def _download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    prefix: str,
    settings: Any,
) -> pd.DataFrame:
    """
    Download one ticker and standardize its columns.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol.

    start_date:
        Earliest requested date.

    end_date:
        Exclusive ending date.

    prefix:
        Prefix placed before every column name.

    settings:
        Quant Machine configuration object.

    Returns
    -------
    pandas.DataFrame
        Cleaned historical data for one ticker.
    """

    print(f"  Downloading {ticker}...")

    try:
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            auto_adjust=settings.auto_adjust_prices,
            progress=False,
            threads=False,
            actions=False,
            repair=False,
            keepna=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"Yahoo Finance download failed for {ticker}: {error}"
        ) from error

    if data.empty:
        raise RuntimeError(
            f"No historical data was returned for ticker {ticker!r}. "
            "Check the ticker symbol, dates, and internet connection."
        )

    data = _flatten_download_columns(data)
    data.index = _remove_timezone(data.index)
    data.index.name = "Date"

    expected_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing_columns = expected_columns.difference(data.columns)

    if missing_columns:
        raise RuntimeError(
            f"Downloaded data for {ticker} is missing these columns: "
            f"{sorted(missing_columns)}"
        )

    selected_columns = [
        column
        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        if column in data.columns
    ]

    data = data[selected_columns].copy()

    rename_map = {
        column: f"{prefix}_{column.lower().replace(' ', '_')}"
        for column in data.columns
    }

    data = data.rename(columns=rename_map)

    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    return data


def download_market_data(settings: Any) -> pd.DataFrame:
    """
    Download the target stock and external market-context datasets.

    The target stock determines the final row index. External context is
    joined to those target-stock trading dates.
    """

    download_end = _calculate_download_end_date(settings)

    print(f"  Download period: {settings.start_date} to {download_end}")
    print("  Note: Yahoo Finance treats the ending date as exclusive.")

    target_data = _download_single_ticker(
        ticker=settings.ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="target",
        settings=settings,
    )

    market_data = _download_single_ticker(
        ticker=settings.market_ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="market",
        settings=settings,
    )

    sector_data = _download_single_ticker(
        ticker=settings.sector_ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="sector",
        settings=settings,
    )

    nasdaq_data = _download_single_ticker(
        ticker=settings.nasdaq_ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="nasdaq",
        settings=settings,
    )

    vix_data = _download_single_ticker(
        ticker=settings.vix_ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="vix",
        settings=settings,
    )

    small_cap_data = _download_single_ticker(
        ticker=settings.small_cap_ticker,
        start_date=settings.start_date,
        end_date=download_end,
        prefix="small_cap",
        settings=settings,
    )

    combined = target_data.join(
        [
            market_data,
            sector_data,
            nasdaq_data,
            vix_data,
            small_cap_data,
        ],
        how="left",
    )

    if settings.forward_fill_market_context:
        context_columns = [
            column
            for column in combined.columns
            if not column.startswith("target_")
        ]

        combined[context_columns] = combined[context_columns].ffill()

    combined = combined.sort_index()

    required_target_columns = [
        "target_open",
        "target_high",
        "target_low",
        "target_close",
        "target_volume",
    ]

    combined = combined.dropna(subset=required_target_columns)

    if len(combined) < settings.minimum_training_rows:
        raise RuntimeError(
            f"Only {len(combined):,} target-stock rows were downloaded. "
            f"The configuration requires at least "
            f"{settings.minimum_training_rows:,} rows."
        )

    print(f"  Target trading rows: {len(combined):,}")
    print(
        f"  Available date range: "
        f"{combined.index.min():%Y-%m-%d} to "
        f"{combined.index.max():%Y-%m-%d}"
    )

    return combined


# ============================================================
# TECHNICAL INDICATOR HELPERS
# ============================================================


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two Series while protecting against zero denominators."""

    cleaned_denominator = denominator.replace(0.0, np.nan)
    return numerator / cleaned_denominator


def _calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate the Relative Strength Index.

    Exponential smoothing is used for average gains and losses.
    """

    price_change = close.diff()

    gains = price_change.clip(lower=0.0)
    losses = -price_change.clip(upper=0.0)

    average_gain = gains.ewm(
        alpha=1.0 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1.0 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = _safe_divide(
        average_gain,
        average_loss,
    )

    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    rsi = rsi.where(average_loss != 0.0, 100.0)
    rsi = rsi.where(average_gain != 0.0, 0.0)

    return rsi


def _calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate daily True Range."""

    previous_close = close.shift(1)

    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def _calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""

    true_range = _calculate_true_range(
        high=high,
        low=low,
        close=close,
    )

    return true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()


def _calculate_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate On-Balance Volume."""

    direction = np.sign(close.diff()).fillna(0.0)
    signed_volume = volume * direction

    return signed_volume.cumsum()


def _calculate_stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate the stochastic oscillator percentage K."""

    rolling_high = high.rolling(
        window=period,
        min_periods=period,
    ).max()

    rolling_low = low.rolling(
        window=period,
        min_periods=period,
    ).min()

    return 100.0 * _safe_divide(
        close - rolling_low,
        rolling_high - rolling_low,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def _add_price_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create target-stock price and return features."""

    close = data["target_close"]
    open_price = data["target_open"]
    high = data["target_high"]
    low = data["target_low"]

    return_periods = [1, 2, 3, 5, 10, 20, 60]

    for period in return_periods:
        column = f"target_return_{period}d"
        data[column] = close.pct_change(
            periods=period,
            fill_method=None,
        )
        feature_columns.append(column)

    data["opening_gap"] = _safe_divide(
        open_price,
        close.shift(1),
    ) - 1.0
    feature_columns.append("opening_gap")

    data["intraday_return"] = _safe_divide(
        close,
        open_price,
    ) - 1.0
    feature_columns.append("intraday_return")

    data["intraday_range"] = _safe_divide(
        high - low,
        close,
    )
    feature_columns.append("intraday_range")

    data["close_location"] = _safe_divide(
        close - low,
        high - low,
    )
    feature_columns.append("close_location")

    data["overnight_to_range"] = _safe_divide(
        open_price - close.shift(1),
        high - low,
    )
    feature_columns.append("overnight_to_range")


def _add_trend_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create moving-average and trend features."""

    close = data["target_close"]

    moving_average_periods = [5, 10, 20, 50, 100, 200]

    moving_averages: dict[int, pd.Series] = {}

    for period in moving_average_periods:
        moving_average = close.rolling(
            window=period,
            min_periods=period,
        ).mean()

        moving_averages[period] = moving_average

        column = f"price_to_sma_{period}"
        data[column] = _safe_divide(
            close,
            moving_average,
        ) - 1.0
        feature_columns.append(column)

    data["sma_5_to_20"] = _safe_divide(
        moving_averages[5],
        moving_averages[20],
    ) - 1.0
    feature_columns.append("sma_5_to_20")

    data["sma_10_to_50"] = _safe_divide(
        moving_averages[10],
        moving_averages[50],
    ) - 1.0
    feature_columns.append("sma_10_to_50")

    data["sma_20_to_100"] = _safe_divide(
        moving_averages[20],
        moving_averages[100],
    ) - 1.0
    feature_columns.append("sma_20_to_100")

    data["sma_50_to_200"] = _safe_divide(
        moving_averages[50],
        moving_averages[200],
    ) - 1.0
    feature_columns.append("sma_50_to_200")

    data["sma_20_slope_5d"] = moving_averages[20].pct_change(
        periods=5,
        fill_method=None,
    )
    feature_columns.append("sma_20_slope_5d")

    data["sma_50_slope_10d"] = moving_averages[50].pct_change(
        periods=10,
        fill_method=None,
    )
    feature_columns.append("sma_50_slope_10d")

    ema_12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    data["macd_to_price"] = _safe_divide(
        macd,
        close,
    )
    feature_columns.append("macd_to_price")

    data["macd_signal_gap"] = _safe_divide(
        macd - macd_signal,
        close,
    )
    feature_columns.append("macd_signal_gap")


def _add_momentum_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create momentum and price-position features."""

    close = data["target_close"]
    high = data["target_high"]
    low = data["target_low"]

    data["rsi_14"] = _calculate_rsi(
        close=close,
        period=14,
    ) / 100.0
    feature_columns.append("rsi_14")

    data["rsi_5"] = _calculate_rsi(
        close=close,
        period=5,
    ) / 100.0
    feature_columns.append("rsi_5")

    data["stochastic_14"] = _calculate_stochastic_oscillator(
        high=high,
        low=low,
        close=close,
        period=14,
    ) / 100.0
    feature_columns.append("stochastic_14")

    for period in [5, 10, 20]:
        column = f"rate_of_change_{period}d"

        data[column] = _safe_divide(
            close,
            close.shift(period),
        ) - 1.0

        feature_columns.append(column)

    rolling_high_20 = high.rolling(
        window=20,
        min_periods=20,
    ).max()

    rolling_low_20 = low.rolling(
        window=20,
        min_periods=20,
    ).min()

    data["distance_from_20d_high"] = _safe_divide(
        close,
        rolling_high_20,
    ) - 1.0
    feature_columns.append("distance_from_20d_high")

    data["distance_from_20d_low"] = _safe_divide(
        close,
        rolling_low_20,
    ) - 1.0
    feature_columns.append("distance_from_20d_low")

    positive_day = (close.diff() > 0.0).astype(int)
    negative_day = (close.diff() < 0.0).astype(int)

    data["up_days_last_5"] = positive_day.rolling(
        window=5,
        min_periods=5,
    ).sum() / 5.0
    feature_columns.append("up_days_last_5")

    data["down_days_last_5"] = negative_day.rolling(
        window=5,
        min_periods=5,
    ).sum() / 5.0
    feature_columns.append("down_days_last_5")

    data["up_days_last_20"] = positive_day.rolling(
        window=20,
        min_periods=20,
    ).sum() / 20.0
    feature_columns.append("up_days_last_20")


def _add_volatility_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create volatility, ATR, and Bollinger Band features."""

    close = data["target_close"]
    high = data["target_high"]
    low = data["target_low"]

    daily_return = close.pct_change(fill_method=None)

    for period in [5, 10, 20, 60]:
        column = f"realized_volatility_{period}d"

        data[column] = daily_return.rolling(
            window=period,
            min_periods=period,
        ).std()

        feature_columns.append(column)

    atr_14 = _calculate_atr(
        high=high,
        low=low,
        close=close,
        period=14,
    )

    data["atr_14_to_price"] = _safe_divide(
        atr_14,
        close,
    )
    feature_columns.append("atr_14_to_price")

    rolling_mean_20 = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    rolling_std_20 = close.rolling(
        window=20,
        min_periods=20,
    ).std()

    upper_band = rolling_mean_20 + (2.0 * rolling_std_20)
    lower_band = rolling_mean_20 - (2.0 * rolling_std_20)

    data["bollinger_position"] = _safe_divide(
        close - lower_band,
        upper_band - lower_band,
    )
    feature_columns.append("bollinger_position")

    data["bollinger_width"] = _safe_divide(
        upper_band - lower_band,
        rolling_mean_20,
    )
    feature_columns.append("bollinger_width")

    data["volatility_ratio_5_to_20"] = _safe_divide(
        data["realized_volatility_5d"],
        data["realized_volatility_20d"],
    )
    feature_columns.append("volatility_ratio_5_to_20")

    data["volatility_ratio_20_to_60"] = _safe_divide(
        data["realized_volatility_20d"],
        data["realized_volatility_60d"],
    )
    feature_columns.append("volatility_ratio_20_to_60")


def _add_volume_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create target-stock trading-volume features."""

    close = data["target_close"]
    volume = data["target_volume"]

    data["volume_change_1d"] = volume.pct_change(
        periods=1,
        fill_method=None,
    )
    feature_columns.append("volume_change_1d")

    data["volume_change_5d"] = volume.pct_change(
        periods=5,
        fill_method=None,
    )
    feature_columns.append("volume_change_5d")

    for period in [5, 20, 60]:
        rolling_volume = volume.rolling(
            window=period,
            min_periods=period,
        ).mean()

        column = f"volume_to_average_{period}d"

        data[column] = _safe_divide(
            volume,
            rolling_volume,
        )

        feature_columns.append(column)

    dollar_volume = close * volume

    data["log_dollar_volume"] = np.log1p(
        dollar_volume.clip(lower=0.0)
    )
    feature_columns.append("log_dollar_volume")

    data["dollar_volume_change_5d"] = dollar_volume.pct_change(
        periods=5,
        fill_method=None,
    )
    feature_columns.append("dollar_volume_change_5d")

    obv = _calculate_obv(
        close=close,
        volume=volume,
    )

    obv_average_20 = obv.rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["obv_to_average_20d"] = _safe_divide(
        obv,
        obv_average_20,
    ) - 1.0
    feature_columns.append("obv_to_average_20d")

    data["return_volume_interaction"] = (
        data["target_return_1d"]
        * data["volume_to_average_20d"]
    )
    feature_columns.append("return_volume_interaction")


def _add_external_market_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create market, sector, Nasdaq, VIX, and small-cap features."""

    external_assets = {
        "market": "market_close",
        "sector": "sector_close",
        "nasdaq": "nasdaq_close",
        "small_cap": "small_cap_close",
    }

    for asset_name, close_column in external_assets.items():
        close = data[close_column]

        for period in [1, 5, 20]:
            feature_name = f"{asset_name}_return_{period}d"

            data[feature_name] = close.pct_change(
                periods=period,
                fill_method=None,
            )

            feature_columns.append(feature_name)

        sma_20 = close.rolling(
            window=20,
            min_periods=20,
        ).mean()

        sma_200 = close.rolling(
            window=200,
            min_periods=200,
        ).mean()

        feature_name_20 = f"{asset_name}_price_to_sma_20"
        feature_name_200 = f"{asset_name}_price_to_sma_200"

        data[feature_name_20] = _safe_divide(
            close,
            sma_20,
        ) - 1.0

        data[feature_name_200] = _safe_divide(
            close,
            sma_200,
        ) - 1.0

        feature_columns.extend(
            [
                feature_name_20,
                feature_name_200,
            ]
        )

    vix_close = data["vix_close"]

    data["vix_log_level"] = np.log1p(
        vix_close.clip(lower=0.0)
    )
    feature_columns.append("vix_log_level")

    data["vix_change_1d"] = vix_close.pct_change(
        periods=1,
        fill_method=None,
    )
    feature_columns.append("vix_change_1d")

    data["vix_change_5d"] = vix_close.pct_change(
        periods=5,
        fill_method=None,
    )
    feature_columns.append("vix_change_5d")

    vix_average_20 = vix_close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["vix_to_average_20d"] = _safe_divide(
        vix_close,
        vix_average_20,
    ) - 1.0
    feature_columns.append("vix_to_average_20d")

    data["relative_to_market_1d"] = (
        data["target_return_1d"]
        - data["market_return_1d"]
    )
    feature_columns.append("relative_to_market_1d")

    data["relative_to_market_5d"] = (
        data["target_return_5d"]
        - data["market_return_5d"]
    )
    feature_columns.append("relative_to_market_5d")

    data["relative_to_sector_1d"] = (
        data["target_return_1d"]
        - data["sector_return_1d"]
    )
    feature_columns.append("relative_to_sector_1d")

    data["relative_to_sector_5d"] = (
        data["target_return_5d"]
        - data["sector_return_5d"]
    )
    feature_columns.append("relative_to_sector_5d")

    data["sector_relative_to_market_5d"] = (
        data["sector_return_5d"]
        - data["market_return_5d"]
    )
    feature_columns.append("sector_relative_to_market_5d")


def _add_market_regime_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Create simple continuous and binary market-regime features."""

    market_close = data["market_close"]
    sector_close = data["sector_close"]
    vix_close = data["vix_close"]

    market_sma_50 = market_close.rolling(
        window=50,
        min_periods=50,
    ).mean()

    market_sma_200 = market_close.rolling(
        window=200,
        min_periods=200,
    ).mean()

    sector_sma_50 = sector_close.rolling(
        window=50,
        min_periods=50,
    ).mean()

    market_returns = market_close.pct_change(fill_method=None)

    market_volatility_20 = market_returns.rolling(
        window=20,
        min_periods=20,
    ).std()

    market_volatility_252_median = market_volatility_20.rolling(
        window=252,
        min_periods=100,
    ).median()

    vix_252_median = vix_close.rolling(
        window=252,
        min_periods=100,
    ).median()

    data["market_above_sma_200"] = (
        market_close > market_sma_200
    ).astype(float)
    feature_columns.append("market_above_sma_200")

    data["market_sma_50_above_200"] = (
        market_sma_50 > market_sma_200
    ).astype(float)
    feature_columns.append("market_sma_50_above_200")

    data["sector_above_sma_50"] = (
        sector_close > sector_sma_50
    ).astype(float)
    feature_columns.append("sector_above_sma_50")

    data["high_market_volatility_regime"] = (
        market_volatility_20 > market_volatility_252_median
    ).astype(float)
    feature_columns.append("high_market_volatility_regime")

    data["high_vix_regime"] = (
        vix_close > vix_252_median
    ).astype(float)
    feature_columns.append("high_vix_regime")

    data["market_trend_strength"] = _safe_divide(
        market_sma_50,
        market_sma_200,
    ) - 1.0
    feature_columns.append("market_trend_strength")


def _add_calendar_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Create cyclical calendar features.

    Calendar variables use the date known at prediction time and therefore
    do not introduce future market information.
    """

    day_of_week = data.index.dayofweek
    month = data.index.month

    data["day_of_week_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 5.0
    )
    feature_columns.append("day_of_week_sin")

    data["day_of_week_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 5.0
    )
    feature_columns.append("day_of_week_cos")

    data["month_sin"] = np.sin(
        2.0 * np.pi * month / 12.0
    )
    feature_columns.append("month_sin")

    data["month_cos"] = np.cos(
        2.0 * np.pi * month / 12.0
    )
    feature_columns.append("month_cos")


def _add_prediction_target(
    data: pd.DataFrame,
    settings: Any,
) -> None:
    """
    Create the future return and binary prediction target.

    These columns are not included in feature_columns.

    A row is labeled 1 when the future return exceeds the configured
    target threshold. It is labeled 0 otherwise.
    """

    horizon = settings.target_horizon
    close = data["target_close"]

    future_close = close.shift(-horizon)

    data["future_close"] = future_close

    data["future_return"] = (
        _safe_divide(
            future_close,
            close,
        )
        - 1.0
    )

    known_future = data["future_return"].notna()

    data["target_up"] = np.nan

    data.loc[known_future, "target_up"] = (
        data.loc[known_future, "future_return"]
        > settings.target_return_threshold
    ).astype(int)


def _validate_feature_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    settings: Any,
) -> None:
    """Validate the completed feature dataset."""

    if not feature_columns:
        raise RuntimeError("No model features were created.")

    duplicate_features = pd.Index(feature_columns).duplicated()

    if duplicate_features.any():
        duplicates = pd.Index(feature_columns)[duplicate_features].tolist()

        raise RuntimeError(
            f"Duplicate feature names were created: {duplicates}"
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Feature columns are missing from the dataset: "
            f"{missing_features}"
        )

    completed_training_rows = data.dropna(
        subset=["target_up"],
    )

    completed_training_rows = completed_training_rows.dropna(
        subset=feature_columns,
        how="all",
    )

    if len(completed_training_rows) < settings.minimum_training_rows:
        raise RuntimeError(
            f"Feature engineering produced only "
            f"{len(completed_training_rows):,} usable training rows. "
            f"At least {settings.minimum_training_rows:,} are required."
        )

    if data.index.has_duplicates:
        raise RuntimeError(
            "The feature dataset contains duplicate trading dates."
        )

    if not data.index.is_monotonic_increasing:
        raise RuntimeError(
            "The feature dataset is not ordered by trading date."
        )


def build_feature_dataset(
    raw_market_data: pd.DataFrame,
    settings: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the complete machine-learning dataset.

    Parameters
    ----------
    raw_market_data:
        Combined target-stock and market-context data.

    settings:
        Quant Machine configuration object.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The complete dataset and the exact list of model feature columns.
    """

    data = raw_market_data.copy()
    data = data.sort_index()

    feature_columns: list[str] = []

    _add_price_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_trend_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_momentum_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_volatility_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_volume_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_external_market_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_market_regime_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_calendar_features(
        data=data,
        feature_columns=feature_columns,
    )

    _add_prediction_target(
        data=data,
        settings=settings,
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = list(dict.fromkeys(feature_columns))

    _validate_feature_dataset(
        data=data,
        feature_columns=feature_columns,
        settings=settings,
    )

    usable_training_rows = data.dropna(
        subset=["target_up"],
    ).dropna(
        subset=feature_columns,
        how="all",
    )

    print(f"  Features created: {len(feature_columns)}")
    print(f"  Usable labeled rows: {len(usable_training_rows):,}")
    print(
        f"  First usable date: "
        f"{usable_training_rows.index.min():%Y-%m-%d}"
    )
    print(
        f"  Latest downloaded date: "
        f"{data.index.max():%Y-%m-%d}"
    )

    return data, feature_columns