"""
Walk-Forward Backtesting
========================

This file connects model training and prediction into two workflows:

1. Historical walk-forward backtesting
2. Live next-trading-day prediction

Historical testing follows this process:

- Train using only outcomes known at that time
- Generate and lock the prediction
- Reveal the historical result
- Record accuracy and trading performance
- Move forward one trading day
- Retrain and repeat
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models import ModelEnsemble
from prediction import (
    PredictionResult,
    make_prediction,
    predictions_to_dataframe,
    print_prediction_result,
)
from training import train_model_ensemble


# ============================================================
# DATE HELPERS
# ============================================================


def _get_latest_date_on_or_before(
    index: pd.DatetimeIndex,
    requested_date: pd.Timestamp | str,
) -> pd.Timestamp:
    """
    Return the latest available trading date on or before a requested date.

    This allows a cutoff date to fall on a weekend or market holiday.
    """

    requested_timestamp = pd.Timestamp(requested_date)

    available_dates = index[index <= requested_timestamp]

    if len(available_dates) == 0:
        raise ValueError(
            f"No market data exists on or before "
            f"{requested_timestamp:%Y-%m-%d}."
        )

    return pd.Timestamp(available_dates[-1])


def _get_index_position(
    index: pd.DatetimeIndex,
    requested_date: pd.Timestamp,
) -> int:
    """Return the integer position of one trading date."""

    location = index.get_loc(requested_date)

    if not isinstance(location, (int, np.integer)):
        raise RuntimeError(
            f"The trading date {requested_date:%Y-%m-%d} "
            "has an ambiguous position."
        )

    return int(location)


def _get_training_label_end_date(
    data: pd.DataFrame,
    information_date: pd.Timestamp,
    target_horizon: int,
) -> pd.Timestamp:
    """
    Find the latest feature row whose target was known by information_date.

    Example with a one-day target
    -----------------------------
    To forecast using July 15 information:

    - July 15's target depends on July 16
    - July 15 cannot be a labeled training row
    - July 14's target depends on July 15
    - July 14 can be used for training

    Therefore, the training-label endpoint is one trading row before the
    information date.
    """

    information_position = _get_index_position(
        index=data.index,
        requested_date=information_date,
    )

    training_end_position = information_position - target_horizon

    if training_end_position < 0:
        raise RuntimeError(
            "There is not enough history before the requested "
            "information date to create known training labels."
        )

    return pd.Timestamp(data.index[training_end_position])


def _get_historical_prediction_dates(
    data: pd.DataFrame,
    settings: Any,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return information-date and forecast-date pairs for a backtest.

    The cutoff date represents the final completed trading day initially
    available to the model.

    For a one-day target:

    cutoff date: July 15
    first forecast: July 16
    first information date: July 15
    """

    cutoff_date = _get_latest_date_on_or_before(
        index=data.index,
        requested_date=settings.cutoff_date,
    )

    cutoff_position = _get_index_position(
        index=data.index,
        requested_date=cutoff_date,
    )

    date_pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for step in range(1, settings.prediction_days + 1):
        forecast_position = cutoff_position + step
        information_position = (
            forecast_position - settings.target_horizon
        )

        if forecast_position >= len(data):
            break

        if information_position < 0:
            continue

        forecast_date = pd.Timestamp(
            data.index[forecast_position]
        )

        information_date = pd.Timestamp(
            data.index[information_position]
        )

        date_pairs.append(
            (
                information_date,
                forecast_date,
            )
        )

    if len(date_pairs) < settings.prediction_days:
        raise RuntimeError(
            f"The dataset contains only {len(date_pairs)} completed "
            f"forecast date(s) after the cutoff, but "
            f"{settings.prediction_days} were requested. "
            "Increase the future download buffer or use an earlier cutoff."
        )

    return date_pairs


def _get_latest_usable_information_date(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Timestamp:
    """
    Return the latest row containing at least one usable model feature.

    The preprocessing pipelines can impute individual missing features, but
    the entire feature row cannot be missing.
    """

    feature_data = data[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    usable_rows = ~feature_data.isna().all(axis=1)

    if not usable_rows.any():
        raise RuntimeError(
            "The dataset contains no usable feature rows."
        )

    return pd.Timestamp(
        feature_data.index[usable_rows][-1]
    )


# ============================================================
# TRADING SIMULATION
# ============================================================


def _signal_to_position(
    signal: str,
    settings: Any,
) -> float:
    """
    Convert a forecast signal into a portfolio position.

    Returns
    -------
    float
        Positive value for a long position, negative value for a short
        position, or zero for no position.
    """

    if signal == "UP" and settings.allow_long_positions:
        return float(settings.position_size)

    if signal == "DOWN" and settings.allow_short_positions:
        return -float(settings.position_size)

    return 0.0


def _calculate_strategy_return(
    signal: str,
    actual_return: float | None,
    settings: Any,
) -> tuple[float, float, float]:
    """
    Calculate position, trading cost, and net strategy return.

    Transaction cost is charged once for each prediction that creates a
    position. This is a simplified assumption appropriate for the first
    version of the project.
    """

    if actual_return is None or not np.isfinite(actual_return):
        return 0.0, 0.0, 0.0

    position = _signal_to_position(
        signal=signal,
        settings=settings,
    )

    if position == 0.0:
        return 0.0, 0.0, 0.0

    gross_return = position * actual_return

    trading_cost = (
        abs(position)
        * settings.transaction_cost
    )

    net_return = gross_return - trading_cost

    return position, trading_cost, net_return


def _add_trading_results(
    results_frame: pd.DataFrame,
    settings: Any,
) -> pd.DataFrame:
    """Add positions, costs, strategy returns, and equity values."""

    if results_frame.empty:
        return results_frame

    output = results_frame.copy()

    positions: list[float] = []
    trading_costs: list[float] = []
    strategy_returns: list[float] = []

    for _, row in output.iterrows():
        position, cost, strategy_return = (
            _calculate_strategy_return(
                signal=str(row["signal"]),
                actual_return=row["actual_return"],
                settings=settings,
            )
        )

        positions.append(position)
        trading_costs.append(cost)
        strategy_returns.append(strategy_return)

    output["position"] = positions
    output["trading_cost"] = trading_costs
    output["strategy_return"] = strategy_returns

    output["strategy_growth"] = (
        1.0 + output["strategy_return"]
    ).cumprod()

    output["strategy_equity"] = (
        settings.initial_capital
        * output["strategy_growth"]
    )

    output["buy_hold_growth"] = (
        1.0 + output["actual_return"].fillna(0.0)
    ).cumprod()

    output["buy_hold_equity"] = (
        settings.initial_capital
        * output["buy_hold_growth"]
    )

    return output


# ============================================================
# PERFORMANCE METRICS
# ============================================================


def _compound_return(returns: pd.Series) -> float:
    """Calculate a compounded return series."""

    clean_returns = returns.dropna()

    if clean_returns.empty:
        return 0.0

    return float(
        (1.0 + clean_returns).prod() - 1.0
    )


def _calculate_sharpe_ratio(
    returns: pd.Series,
    target_horizon: int,
) -> float | None:
    """
    Calculate an annualized zero-risk-rate Sharpe ratio.

    This is meaningful only with a sufficiently long test. A five-day
    backtest can calculate it mathematically, but the result is not reliable.
    """

    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return None

    standard_deviation = clean_returns.std(ddof=1)

    if standard_deviation == 0 or not np.isfinite(
        standard_deviation
    ):
        return None

    periods_per_year = 252.0 / target_horizon

    sharpe = (
        clean_returns.mean()
        / standard_deviation
        * np.sqrt(periods_per_year)
    )

    return float(sharpe)


def _calculate_max_drawdown(
    growth_series: pd.Series,
) -> float:
    """Calculate the maximum peak-to-trough equity decline."""

    clean_growth = growth_series.dropna()

    if clean_growth.empty:
        return 0.0

    running_peak = clean_growth.cummax()

    drawdown = (
        clean_growth / running_peak
    ) - 1.0

    return float(drawdown.min())


def _calculate_brier_score(
    results_frame: pd.DataFrame,
) -> float | None:
    """
    Calculate the Brier score for probability accuracy.

    A lower Brier score is better. It evaluates whether predicted
    probabilities correspond with actual binary outcomes.
    """

    required_columns = {
        "probability_up",
        "actual_direction",
    }

    if not required_columns.issubset(results_frame.columns):
        return None

    valid_rows = results_frame[
        results_frame["actual_direction"].isin(
            ["UP", "DOWN"]
        )
    ].copy()

    if valid_rows.empty:
        return None

    actual_up = (
        valid_rows["actual_direction"] == "UP"
    ).astype(float)

    squared_errors = (
        valid_rows["probability_up"] - actual_up
    ) ** 2

    return float(squared_errors.mean())


def _calculate_performance_metrics(
    results_frame: pd.DataFrame,
    settings: Any,
) -> dict[str, Any]:
    """Calculate classification and trading performance statistics."""

    if results_frame.empty:
        raise RuntimeError(
            "Cannot calculate metrics from an empty backtest."
        )

    scored_predictions = results_frame[
        results_frame["correct"].notna()
    ]

    trades = results_frame[
        results_frame["position"] != 0.0
    ]

    winning_trades = trades[
        trades["strategy_return"] > 0.0
    ]

    losing_trades = trades[
        trades["strategy_return"] < 0.0
    ]

    accuracy: float | None

    if scored_predictions.empty:
        accuracy = None
    else:
        accuracy = float(
            scored_predictions["correct"].astype(bool).mean()
        )

    trade_count = int(len(trades))

    win_rate: float | None

    if trade_count == 0:
        win_rate = None
    else:
        win_rate = float(
            len(winning_trades) / trade_count
        )

    average_trade_return: float | None

    if trade_count == 0:
        average_trade_return = None
    else:
        average_trade_return = float(
            trades["strategy_return"].mean()
        )

    average_win: float | None

    if winning_trades.empty:
        average_win = None
    else:
        average_win = float(
            winning_trades["strategy_return"].mean()
        )

    average_loss: float | None

    if losing_trades.empty:
        average_loss = None
    else:
        average_loss = float(
            losing_trades["strategy_return"].mean()
        )

    strategy_return = _compound_return(
        results_frame["strategy_return"]
    )

    buy_hold_return = _compound_return(
        results_frame["actual_return"]
    )

    sharpe_ratio = _calculate_sharpe_ratio(
        returns=results_frame["strategy_return"],
        target_horizon=settings.target_horizon,
    )

    max_drawdown = _calculate_max_drawdown(
        growth_series=results_frame["strategy_growth"],
    )

    brier_score = _calculate_brier_score(
        results_frame=results_frame,
    )

    final_equity = (
        settings.initial_capital
        * (1.0 + strategy_return)
    )

    return {
        "accuracy": accuracy,
        "prediction_count": int(len(results_frame)),
        "scored_prediction_count": int(
            len(scored_predictions)
        ),
        "hold_count": int(
            (results_frame["signal"] == "HOLD").sum()
        ),
        "trade_count": trade_count,
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "average_win": average_win,
        "average_loss": average_loss,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "brier_score": brier_score,
        "final_equity": final_equity,
        "results": results_frame,
    }


# ============================================================
# REPORTING
# ============================================================


def _format_optional_percentage(
    value: float | None,
) -> str:
    """Format an optional decimal value as a percentage."""

    if value is None:
        return "N/A"

    return f"{value:.2%}"


def _format_optional_number(
    value: float | None,
    decimal_places: int = 2,
) -> str:
    """Format an optional numerical value."""

    if value is None:
        return "N/A"

    return f"{value:.{decimal_places}f}"


def _print_backtest_summary(
    metrics: dict[str, Any],
) -> None:
    """Display complete historical-test results."""

    print("\n" + "=" * 68)
    print("BACKTEST RESULTS".center(68))
    print("=" * 68)

    print(
        f"Predictions evaluated : "
        f"{metrics['prediction_count']}"
    )

    print(
        f"Directional forecasts : "
        f"{metrics['scored_prediction_count']}"
    )

    print(
        f"HOLD signals          : "
        f"{metrics['hold_count']}"
    )

    print(
        f"Directional accuracy  : "
        f"{_format_optional_percentage(metrics['accuracy'])}"
    )

    print(
        f"Brier score            : "
        f"{_format_optional_number(metrics['brier_score'], 4)}"
    )

    print("-" * 68)

    print(
        f"Trades taken           : "
        f"{metrics['trade_count']}"
    )

    print(
        f"Trade win rate         : "
        f"{_format_optional_percentage(metrics['win_rate'])}"
    )

    print(
        f"Average trade return   : "
        f"{_format_optional_percentage(
            metrics['average_trade_return']
        )}"
    )

    print(
        f"Strategy return        : "
        f"{metrics['strategy_return']:+.2%}"
    )

    print(
        f"Buy-and-hold return    : "
        f"{metrics['buy_hold_return']:+.2%}"
    )

    print(
        f"Maximum drawdown       : "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Annualized Sharpe      : "
        f"{_format_optional_number(
            metrics['sharpe_ratio'],
            2,
        )}"
    )

    print(
        f"Final strategy equity  : "
        f"${metrics['final_equity']:,.2f}"
    )

    print("=" * 68)

    if metrics["prediction_count"] < 30:
        print(
            "\nWarning: This is a very small test sample. "
            "Accuracy, Sharpe ratio, and win rate may be misleading."
        )


def _save_backtest_results(
    results_frame: pd.DataFrame,
    settings: Any,
) -> Path:
    """Save detailed historical predictions to a CSV file."""

    output_path = Path(
        settings.backtest_output_file
    )

    results_frame.to_csv(
        output_path,
        index=False,
    )

    return output_path.resolve()


# ============================================================
# HISTORICAL WALK-FORWARD TEST
# ============================================================


def run_historical_backtest(
    data: pd.DataFrame,
    feature_columns: list[str],
    model: ModelEnsemble,
    settings: Any,
) -> dict[str, Any]:
    """
    Run a realistic day-by-day historical backtest.

    At every step:

    1. Identify the information date
    2. Stop training before any unknown labels
    3. Train fresh models
    4. Generate the prediction
    5. Reveal the known historical result
    6. Move forward and repeat
    """

    date_pairs = _get_historical_prediction_dates(
        data=data,
        settings=settings,
    )

    cutoff_used = _get_latest_date_on_or_before(
        index=data.index,
        requested_date=settings.cutoff_date,
    )

    print(
        f"\n  Historical cutoff used: "
        f"{cutoff_used:%Y-%m-%d}"
    )

    print(
        f"  Walk-forward predictions: "
        f"{len(date_pairs)}"
    )

    prediction_results: list[PredictionResult] = []

    fitted_ensemble = None
    previous_training_end_date = None

    for prediction_number, (
        information_date,
        forecast_date,
    ) in enumerate(
        date_pairs,
        start=1,
    ):
        print(
            f"\nPrediction "
            f"{prediction_number}/{len(date_pairs)}"
        )

        print(
            f"  Information date: "
            f"{information_date:%Y-%m-%d}"
        )

        print(
            f"  Forecast date   : "
            f"{forecast_date:%Y-%m-%d}"
        )

        training_label_end_date = (
            _get_training_label_end_date(
                data=data,
                information_date=information_date,
                target_horizon=settings.target_horizon,
            )
        )

        should_retrain = (
            fitted_ensemble is None
            or settings.retrain_every_step
            or previous_training_end_date
            != training_label_end_date
        )

        if should_retrain:
            fitted_ensemble = train_model_ensemble(
                data=data,
                feature_columns=feature_columns,
                model=model,
                settings=settings,
                training_end_date=training_label_end_date,
            )

            previous_training_end_date = (
                training_label_end_date
            )

        if fitted_ensemble is None:
            raise RuntimeError(
                "The model ensemble was not trained."
            )

        # The probability and signal are generated before the historical
        # outcome is attached to the result.
        result = make_prediction(
            data=data,
            feature_columns=feature_columns,
            fitted_ensemble=fitted_ensemble,
            information_date=information_date,
            settings=settings,
            reveal_actual=True,
        )

        if result.forecast_date != forecast_date:
            raise RuntimeError(
                "Calculated forecast date does not match the "
                "walk-forward schedule."
            )

        prediction_results.append(result)

        print_prediction_result(
            result=result,
            settings=settings,
        )

    results_frame = predictions_to_dataframe(
        results=prediction_results,
    )

    results_frame = _add_trading_results(
        results_frame=results_frame,
        settings=settings,
    )

    metrics = _calculate_performance_metrics(
        results_frame=results_frame,
        settings=settings,
    )

    _print_backtest_summary(
        metrics=metrics,
    )

    if settings.save_backtest_csv:
        output_path = _save_backtest_results(
            results_frame=results_frame,
            settings=settings,
        )

        print(
            f"\nDetailed results saved to:\n"
            f"{output_path}"
        )

        metrics["output_file"] = str(output_path)

    return metrics


# ============================================================
# LIVE PREDICTION
# ============================================================


def run_live_prediction(
    data: pd.DataFrame,
    feature_columns: list[str],
    model: ModelEnsemble,
    settings: Any,
) -> dict[str, Any]:
    """
    Train through the latest known outcomes and forecast the next session.

    No actual result is revealed because the future trading session has not
    happened yet.
    """

    information_date = _get_latest_usable_information_date(
        data=data,
        feature_columns=feature_columns,
    )

    training_label_end_date = _get_training_label_end_date(
        data=data,
        information_date=information_date,
        target_horizon=settings.target_horizon,
    )

    print(
        f"\n  Latest completed information date: "
        f"{information_date:%Y-%m-%d}"
    )

    print(
        f"  Latest known training label date : "
        f"{training_label_end_date:%Y-%m-%d}"
    )

    fitted_ensemble = train_model_ensemble(
        data=data,
        feature_columns=feature_columns,
        model=model,
        settings=settings,
        training_end_date=training_label_end_date,
    )

    result = make_prediction(
        data=data,
        feature_columns=feature_columns,
        fitted_ensemble=fitted_ensemble,
        information_date=information_date,
        settings=settings,
        reveal_actual=False,
    )

    print_prediction_result(
        result=result,
        settings=settings,
    )

    return {
        "prediction": result.signal,
        "probability_up": result.probability_up,
        "probability_down": result.probability_down,
        "confidence": result.confidence,
        "latest_date": result.information_date,
        "forecast_date": result.forecast_date,
        "model_probabilities": (
            result.model_probabilities
        ),
        "result": result,
    }