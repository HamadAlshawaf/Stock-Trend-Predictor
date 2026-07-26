"""
Quant Machine
=============

Main entry point for the quantitative stock prediction project.

This file coordinates the full workflow:

1. Read user settings
2. Download and prepare market data
3. Build machine-learning features
4. Create the model ensemble
5. Run either:
   - Historical walk-forward testing
   - Live next-trading-day prediction

Most settings should be changed in user_inputs.py, not here.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from backtest import run_historical_backtest, run_live_prediction
from data_features import build_feature_dataset, download_market_data
from models import create_model_ensemble
from user_inputs import SETTINGS


PROGRAM_NAME = "QUANT MACHINE"
PROGRAM_VERSION = "1.0"


def print_header() -> None:
    """Display the program title and current user configuration."""

    print("\n" + "=" * 68)
    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION}".center(68))
    print("=" * 68)

    print(f"Ticker              : {SETTINGS.ticker}")
    print(f"Mode                : {SETTINGS.mode.upper()}")
    print(f"Training start      : {SETTINGS.start_date}")

    if SETTINGS.mode == "historical":
        print(f"Historical cutoff   : {SETTINGS.cutoff_date}")
        print(f"Prediction days     : {SETTINGS.prediction_days}")
    else:
        print("Historical cutoff   : Latest completed trading day")
        print("Prediction horizon  : Next trading day")

    print(f"Target horizon      : {SETTINGS.target_horizon} trading day(s)")
    print(f"Long threshold      : {SETTINGS.long_probability_threshold:.1%}")
    print(f"Transaction cost    : {SETTINGS.transaction_cost:.3%}")
    print("=" * 68)


def validate_configuration() -> None:
    """Check the user settings before downloading or training anything."""

    valid_modes = {"historical", "live"}

    if SETTINGS.mode not in valid_modes:
        raise ValueError(
            f"MODE must be one of {sorted(valid_modes)}, "
            f"not {SETTINGS.mode!r}."
        )

    if not SETTINGS.ticker.strip():
        raise ValueError("TICKER cannot be empty.")

    if SETTINGS.prediction_days < 1:
        raise ValueError("PREDICTION_DAYS must be at least 1.")

    if SETTINGS.target_horizon < 1:
        raise ValueError("TARGET_HORIZON must be at least 1.")

    if SETTINGS.minimum_training_rows < 200:
        raise ValueError(
            "MINIMUM_TRAINING_ROWS should be at least 200 "
            "to reduce unstable model behavior."
        )

    if not 0.50 <= SETTINGS.long_probability_threshold <= 1.00:
        raise ValueError(
            "LONG_PROBABILITY_THRESHOLD must be between 0.50 and 1.00."
        )

    if not 0.00 <= SETTINGS.transaction_cost <= 0.10:
        raise ValueError(
            "TRANSACTION_COST must be between 0.00 and 0.10."
        )

    if SETTINGS.initial_capital <= 0:
        raise ValueError("INITIAL_CAPITAL must be greater than zero.")

    if SETTINGS.position_size <= 0 or SETTINGS.position_size > 1:
        raise ValueError("POSITION_SIZE must be greater than 0 and at most 1.")

    if SETTINGS.mode == "historical" and not SETTINGS.cutoff_date:
        raise ValueError(
            "CUTOFF_DATE is required when MODE is 'historical'."
        )


def print_step(step_number: int, total_steps: int, message: str) -> None:
    """Print a consistently formatted workflow step."""

    print(f"\n[{step_number}/{total_steps}] {message}")


def print_success(message: str) -> None:
    """Print a successful workflow message."""

    print(f"✓ {message}")


def print_failure(message: str) -> None:
    """Print a failed workflow message."""

    print(f"✗ {message}")


def run_quant_machine() -> dict[str, Any]:
    """
    Execute the complete quant workflow.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the final prediction or backtest results.
    """

    total_steps = 4

    print_step(1, total_steps, "Downloading market data")

    raw_market_data = download_market_data(SETTINGS)

    print_success(
        f"Downloaded and merged {len(raw_market_data):,} trading rows"
    )

    print_step(2, total_steps, "Building quantitative features")

    feature_data, feature_columns = build_feature_dataset(
        raw_market_data,
        SETTINGS,
    )

    print_success(
        f"Created {len(feature_columns)} model features "
        f"across {len(feature_data):,} rows"
    )

    print_step(3, total_steps, "Creating model ensemble")

    model = create_model_ensemble(SETTINGS)

    model_names = getattr(model, "model_names", None)

    if model_names:
        print_success(
            "Loaded models: " + ", ".join(model_names)
        )
    else:
        print_success("Model ensemble created")

    print_step(4, total_steps, "Running prediction workflow")

    if SETTINGS.mode == "historical":
        results = run_historical_backtest(
            data=feature_data,
            feature_columns=feature_columns,
            model=model,
            settings=SETTINGS,
        )
    else:
        results = run_live_prediction(
            data=feature_data,
            feature_columns=feature_columns,
            model=model,
            settings=SETTINGS,
        )

    return results


def print_completion_summary(
    results: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    """Display the final program completion message."""

    print("\n" + "=" * 68)
    print("RUN COMPLETE".center(68))
    print("=" * 68)

    if SETTINGS.mode == "historical":
        accuracy = results.get("accuracy")
        trade_count = results.get("trade_count")
        strategy_return = results.get("strategy_return")
        buy_hold_return = results.get("buy_hold_return")

        if accuracy is not None:
            print(f"Directional accuracy : {accuracy:.2%}")

        if trade_count is not None:
            print(f"Trades taken         : {trade_count}")

        if strategy_return is not None:
            print(f"Strategy return      : {strategy_return:+.2%}")

        if buy_hold_return is not None:
            print(f"Buy-and-hold return  : {buy_hold_return:+.2%}")

    else:
        prediction = results.get("prediction")
        probability_up = results.get("probability_up")
        latest_date = results.get("latest_date")

        if latest_date is not None:
            print(f"Latest market date   : {latest_date}")

        if prediction is not None:
            print(f"Next-day prediction  : {prediction}")

        if probability_up is not None:
            print(f"Probability of UP    : {probability_up:.2%}")

    print(f"Total run time       : {elapsed_seconds:.2f} seconds")
    print("=" * 68)


def main() -> None:
    """Main program entry point."""

    start_time = time.perf_counter()

    try:
        validate_configuration()
        print_header()

        results = run_quant_machine()

        elapsed_seconds = time.perf_counter() - start_time
        print_completion_summary(results, elapsed_seconds)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by the user.")
        sys.exit(1)

    except ModuleNotFoundError as error:
        print_failure("A required Python package or project file is missing.")
        print(f"Details: {error}")
        print(
            "\nAfter requirements.txt is created, run:\n"
            "python -m pip install -r requirements.txt"
        )
        sys.exit(1)

    except ValueError as error:
        print_failure("The project configuration is invalid.")
        print(f"Details: {error}")
        sys.exit(1)

    except RuntimeError as error:
        print_failure("The quant workflow could not complete.")
        print(f"Details: {error}")
        sys.exit(1)

    except Exception as error:
        print_failure("An unexpected error occurred.")
        print(f"Error type: {type(error).__name__}")
        print(f"Details: {error}")

        if getattr(SETTINGS, "debug", False):
            raise

        print(
            "\nSet DEBUG = True in user_inputs.py "
            "to display the complete traceback."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()