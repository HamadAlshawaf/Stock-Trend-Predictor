"""
Prediction Logic
================

This file converts fitted model outputs into a final ensemble prediction.

It handles:

- Individual model probabilities
- Weighted ensemble probability
- UP, DOWN, or HOLD signals
- Prediction confidence
- Historical outcome comparison
- Probability details for later calibration analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training import (
    FittedEnsemble,
    prepare_prediction_features,
)


# ============================================================
# PREDICTION RESULT CONTAINER
# ============================================================


@dataclass
class PredictionResult:
    """
    Store the complete result for one prediction date.

    Parameters
    ----------
    information_date:
        Latest market date whose information was used by the model.

    forecast_date:
        Trading date whose price direction is being predicted.

    probability_up:
        Weighted ensemble probability of an upward move.

    probability_down:
        Weighted ensemble probability of a downward move.

    signal:
        Final decision: UP, DOWN, or HOLD.

    confidence:
        Confidence associated with the final signal.

    model_probabilities:
        Individual probability produced by each fitted model.

    actual_direction:
        Historical result, when available.

    actual_return:
        Historical future return, when available.

    correct:
        Whether the directional prediction was correct.
        HOLD signals are not counted as correct or incorrect.
    """

    information_date: pd.Timestamp
    forecast_date: pd.Timestamp | None
    probability_up: float
    probability_down: float
    signal: str
    confidence: float
    model_probabilities: dict[str, float]
    actual_direction: str | None = None
    actual_return: float | None = None
    correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a flat dictionary."""

        result: dict[str, Any] = {
            "information_date": self.information_date,
            "forecast_date": self.forecast_date,
            "probability_up": self.probability_up,
            "probability_down": self.probability_down,
            "signal": self.signal,
            "confidence": self.confidence,
            "actual_direction": self.actual_direction,
            "actual_return": self.actual_return,
            "correct": self.correct,
        }

        for model_name, probability in self.model_probabilities.items():
            safe_name = (
                model_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            result[f"{safe_name}_probability_up"] = probability

        return result


# ============================================================
# VALIDATION HELPERS
# ============================================================


def _validate_probability(
    probability: float,
    model_name: str,
) -> float:
    """Validate and safely constrain a model probability."""

    if not np.isfinite(probability):
        raise RuntimeError(
            f"{model_name} produced a non-finite probability."
        )

    tolerance = 1e-9

    if probability < -tolerance or probability > 1.0 + tolerance:
        raise RuntimeError(
            f"{model_name} produced an invalid probability: "
            f"{probability}"
        )

    return float(np.clip(probability, 0.0, 1.0))


def _validate_prediction_row(
    feature_row: pd.DataFrame,
    fitted_ensemble: FittedEnsemble,
) -> None:
    """Validate the prediction feature row."""

    if len(feature_row) != 1:
        raise ValueError(
            "Prediction requires exactly one feature row."
        )

    missing_columns = [
        column
        for column in fitted_ensemble.feature_columns
        if column not in feature_row.columns
    ]

    if missing_columns:
        raise ValueError(
            "The prediction row is missing these features: "
            f"{missing_columns}"
        )


# ============================================================
# MODEL PROBABILITIES
# ============================================================


def predict_individual_probabilities(
    fitted_ensemble: FittedEnsemble,
    feature_row: pd.DataFrame,
) -> dict[str, float]:
    """
    Generate an upward-move probability from every fitted model.

    Parameters
    ----------
    fitted_ensemble:
        Trained models created in training.py.

    feature_row:
        One row containing the model features.

    Returns
    -------
    dict[str, float]
        Probability of UP from each model.
    """

    _validate_prediction_row(
        feature_row=feature_row,
        fitted_ensemble=fitted_ensemble,
    )

    probabilities: dict[str, float] = {}

    for model_name, fitted_model in fitted_ensemble.models.items():
        try:
            probability_up = fitted_model.predict_probability_up(
                feature_row
            )

        except Exception as error:
            raise RuntimeError(
                f"Prediction failed for {model_name}: {error}"
            ) from error

        probabilities[model_name] = _validate_probability(
            probability=probability_up,
            model_name=model_name,
        )

    if not probabilities:
        raise RuntimeError(
            "No individual model probabilities were produced."
        )

    return probabilities


def calculate_ensemble_probability(
    fitted_ensemble: FittedEnsemble,
    model_probabilities: dict[str, float],
) -> float:
    """
    Calculate the weighted ensemble probability of an upward move.
    """

    normalized_weights = fitted_ensemble.normalized_weights()

    missing_probabilities = [
        model_name
        for model_name in normalized_weights
        if model_name not in model_probabilities
    ]

    if missing_probabilities:
        raise RuntimeError(
            "Missing model probabilities for: "
            f"{missing_probabilities}"
        )

    weighted_probability = sum(
        model_probabilities[model_name]
        * normalized_weights[model_name]
        for model_name in normalized_weights
    )

    return _validate_probability(
        probability=weighted_probability,
        model_name="Ensemble",
    )


# ============================================================
# SIGNAL CREATION
# ============================================================


def determine_signal(
    probability_up: float,
    fitted_ensemble: FittedEnsemble,
    settings: Any,
) -> tuple[str, float]:
    """
    Convert an ensemble probability into UP, DOWN, or HOLD.

    Rules
    -----
    UP:
        Probability is at least the long threshold and long positions
        are enabled.

    DOWN:
        Probability is at or below the short threshold.

        A DOWN directional forecast may still be reported when short
        trading is disabled. The backtester will decide whether an actual
        short position is allowed.

    HOLD:
        Probability lies between the long and short thresholds.
    """

    probability_down = 1.0 - probability_up

    if (
        settings.allow_long_positions
        and probability_up
        >= fitted_ensemble.probability_threshold
    ):
        return "UP", probability_up

    if (
        probability_up
        <= fitted_ensemble.short_probability_threshold
    ):
        return "DOWN", probability_down

    return "HOLD", max(
        probability_up,
        probability_down,
    )


def determine_actual_direction(
    actual_return: float | None,
    target_return_threshold: float,
) -> str | None:
    """Convert a realized future return into UP or DOWN."""

    if actual_return is None:
        return None

    if not np.isfinite(actual_return):
        return None

    if actual_return > target_return_threshold:
        return "UP"

    return "DOWN"


def determine_correctness(
    signal: str,
    actual_direction: str | None,
) -> bool | None:
    """
    Determine whether a directional forecast was correct.

    HOLD signals are excluded from directional accuracy.
    """

    if actual_direction is None:
        return None

    if signal == "HOLD":
        return None

    return signal == actual_direction


# ============================================================
# FORECAST-DATE HELPERS
# ============================================================


def get_forecast_date(
    data: pd.DataFrame,
    information_date: pd.Timestamp | str,
    target_horizon: int,
) -> pd.Timestamp | None:
    """
    Find the forecast date associated with an information date.

    For historical rows, this returns the trading date target_horizon
    rows later.

    For the latest live row, the next future trading date may not yet be
    available in the downloaded dataset, so None is returned.
    """

    timestamp = pd.Timestamp(information_date)

    if timestamp not in data.index:
        raise ValueError(
            f"Information date {timestamp:%Y-%m-%d} "
            "does not exist in the dataset."
        )

    position = data.index.get_loc(timestamp)

    if not isinstance(position, (int, np.integer)):
        raise RuntimeError(
            "The dataset contains an ambiguous date index."
        )

    forecast_position = int(position) + target_horizon

    if forecast_position >= len(data):
        return None

    return pd.Timestamp(data.index[forecast_position])


def get_actual_return(
    data: pd.DataFrame,
    information_date: pd.Timestamp | str,
) -> float | None:
    """
    Return the stored future return for one information date.

    The value is used only after the prediction has already been created.
    """

    timestamp = pd.Timestamp(information_date)

    if timestamp not in data.index:
        raise ValueError(
            f"Information date {timestamp:%Y-%m-%d} "
            "does not exist in the dataset."
        )

    value = data.at[timestamp, "future_return"]

    if pd.isna(value):
        return None

    return float(value)


# ============================================================
# COMPLETE PREDICTION FUNCTIONS
# ============================================================


def make_prediction(
    data: pd.DataFrame,
    feature_columns: list[str],
    fitted_ensemble: FittedEnsemble,
    information_date: pd.Timestamp | str,
    settings: Any,
    reveal_actual: bool = False,
) -> PredictionResult:
    """
    Create one complete ensemble prediction.

    Parameters
    ----------
    data:
        Complete feature dataset.

    feature_columns:
        Exact model feature columns.

    fitted_ensemble:
        Models trained using information available before the forecast.

    information_date:
        Date whose completed market data is used to make the forecast.

        For example, July 11 data may be used to forecast the next
        trading day.

    settings:
        Quant Machine configuration.

    reveal_actual:
        When True, attach the known historical outcome after generating
        the prediction.

        When False, no actual result is included. Live predictions should
        always use False.

    Returns
    -------
    PredictionResult
        Full ensemble result with individual model probabilities.
    """

    information_timestamp = pd.Timestamp(information_date)

    feature_row = prepare_prediction_features(
        data=data,
        feature_columns=feature_columns,
        prediction_date=information_timestamp,
    )

    model_probabilities = predict_individual_probabilities(
        fitted_ensemble=fitted_ensemble,
        feature_row=feature_row,
    )

    probability_up = calculate_ensemble_probability(
        fitted_ensemble=fitted_ensemble,
        model_probabilities=model_probabilities,
    )

    probability_down = 1.0 - probability_up

    signal, confidence = determine_signal(
        probability_up=probability_up,
        fitted_ensemble=fitted_ensemble,
        settings=settings,
    )

    forecast_date = get_forecast_date(
        data=data,
        information_date=information_timestamp,
        target_horizon=settings.target_horizon,
    )

    actual_return: float | None = None
    actual_direction: str | None = None
    correct: bool | None = None

    if reveal_actual:
        actual_return = get_actual_return(
            data=data,
            information_date=information_timestamp,
        )

        actual_direction = determine_actual_direction(
            actual_return=actual_return,
            target_return_threshold=(
                settings.target_return_threshold
            ),
        )

        correct = determine_correctness(
            signal=signal,
            actual_direction=actual_direction,
        )

    return PredictionResult(
        information_date=information_timestamp,
        forecast_date=forecast_date,
        probability_up=probability_up,
        probability_down=probability_down,
        signal=signal,
        confidence=confidence,
        model_probabilities=model_probabilities,
        actual_direction=actual_direction,
        actual_return=actual_return,
        correct=correct,
    )


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================


def print_prediction_result(
    result: PredictionResult,
    settings: Any,
) -> None:
    """Print one prediction in a readable terminal format."""

    print("\n" + "-" * 68)

    if result.forecast_date is not None:
        print(
            f"Forecast date       : "
            f"{result.forecast_date:%Y-%m-%d}"
        )
    else:
        print("Forecast date       : Next trading day")

    print(
        f"Information through : "
        f"{result.information_date:%Y-%m-%d}"
    )

    print(f"Prediction          : {result.signal}")
    print(f"Probability of UP   : {result.probability_up:.2%}")
    print(f"Probability of DOWN : {result.probability_down:.2%}")
    print(f"Signal confidence   : {result.confidence:.2%}")

    if settings.show_individual_model_predictions:
        print("\nIndividual models:")

        for model_name, probability in (
            result.model_probabilities.items()
        ):
            print(
                f"  {model_name:<24} "
                f"{probability:.2%} UP"
            )

    if result.actual_direction is not None:
        print(f"\nActual direction    : {result.actual_direction}")

    if result.actual_return is not None:
        print(f"Actual return       : {result.actual_return:+.2%}")

    if result.correct is True:
        print("Result              : CORRECT")

    elif result.correct is False:
        print("Result              : INCORRECT")

    elif (
        result.actual_direction is not None
        and result.signal == "HOLD"
    ):
        print("Result              : NOT SCORED (HOLD)")

    print("-" * 68)


def predictions_to_dataframe(
    results: list[PredictionResult],
) -> pd.DataFrame:
    """
    Convert multiple prediction results into a DataFrame.

    This will later be used by backtest.py for CSV exports and
    performance calculations.
    """

    if not results:
        return pd.DataFrame()

    records = [
        result.to_dict()
        for result in results
    ]

    output = pd.DataFrame(records)

    if "forecast_date" in output.columns:
        output = output.sort_values(
            by=["forecast_date", "information_date"],
            na_position="last",
        )

    output = output.reset_index(drop=True)

    return output