"""
Model Training
==============

This file trains fresh copies of every model in the ensemble.

It handles:

- Selecting valid historical training rows
- Preventing future-data leakage
- Replacing missing feature values
- Scaling features when required
- Training each model independently
- Returning a fitted ensemble ready for prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import ModelDefinition, ModelEnsemble


# ============================================================
# FITTED MODEL CONTAINERS
# ============================================================


@dataclass
class FittedModel:
    """
    Store one trained model and its metadata.

    Parameters
    ----------
    name:
        Human-readable model name.

    pipeline:
        Fitted scikit-learn pipeline containing preprocessing and estimator.

    weight:
        Configured ensemble weight.

    training_rows:
        Number of rows used to train this model.

    positive_class_rate:
        Percentage of training labels equal to 1.
    """

    name: str
    pipeline: Pipeline
    weight: float
    training_rows: int
    positive_class_rate: float

    def predict_probability_up(
        self,
        feature_row: pd.DataFrame,
    ) -> float:
        """
        Predict the probability of an upward price move.

        Parameters
        ----------
        feature_row:
            One or more rows containing the model features.

        Returns
        -------
        float
            Probability associated with class 1.
        """

        probabilities = self.pipeline.predict_proba(feature_row)

        estimator = self.pipeline.named_steps["model"]
        classes = list(estimator.classes_)

        if 1 not in classes:
            raise RuntimeError(
                f"{self.name} was trained without class 1."
            )

        positive_class_index = classes.index(1)

        return float(probabilities[0, positive_class_index])


@dataclass
class FittedEnsemble:
    """
    Store all fitted models from one training period.

    A new FittedEnsemble should normally be created for each walk-forward
    prediction date.
    """

    models: dict[str, FittedModel]
    feature_columns: list[str]
    training_start_date: pd.Timestamp
    training_end_date: pd.Timestamp
    training_rows: int
    probability_threshold: float
    short_probability_threshold: float

    @property
    def model_names(self) -> list[str]:
        """Return fitted model names."""

        return list(self.models.keys())

    def normalized_weights(self) -> dict[str, float]:
        """Return fitted model weights normalized to sum to 1.0."""

        total_weight = sum(
            fitted_model.weight
            for fitted_model in self.models.values()
        )

        if total_weight <= 0:
            raise RuntimeError(
                "The fitted ensemble has no positive model weight."
            )

        return {
            model_name: fitted_model.weight / total_weight
            for model_name, fitted_model in self.models.items()
        }


# ============================================================
# TRAINING DATA PREPARATION
# ============================================================


def _validate_feature_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Ensure every expected feature exists in the dataset."""

    if not feature_columns:
        raise ValueError("The feature-column list cannot be empty.")

    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The training dataset is missing these features: "
            f"{missing_columns}"
        )


def _prepare_training_data(
    data: pd.DataFrame,
    feature_columns: list[str],
    settings: Any,
    training_end_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select historical labeled rows available by the training date.

    A row may only be used when its future target outcome is already known by
    the requested training endpoint.

    For a one-day target, a row dated July 15 uses the July 16 closing price
    for its label. Therefore, when predicting July 16, the July 15 row cannot
    be included as a labeled training example because its outcome was not yet
    known at prediction time.

    The newest target_horizon rows before the prediction date are consequently
    excluded from model training.
    """

    _validate_feature_columns(
        data=data,
        feature_columns=feature_columns,
    )

    training_data = data.copy()
    training_data = training_data.sort_index()

    if training_end_date is not None:
        endpoint = pd.Timestamp(training_end_date)

        training_data = training_data.loc[
            training_data.index <= endpoint
        ].copy()

    training_data = training_data.dropna(
        subset=["target_up"],
    )

    training_data = training_data.dropna(
        subset=feature_columns,
        how="all",
    )

    if training_data.empty:
        raise RuntimeError(
            "No labeled historical rows are available for training."
        )

    features = training_data[feature_columns].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    target = training_data["target_up"].astype(int).copy()

    if len(features) < settings.minimum_training_rows:
        raise RuntimeError(
            f"Only {len(features):,} valid training rows are available. "
            f"At least {settings.minimum_training_rows:,} are required."
        )

    unique_classes = sorted(target.unique().tolist())

    if unique_classes != [0, 1]:
        raise RuntimeError(
            "Training requires both DOWN and UP examples. "
            f"Available classes: {unique_classes}"
        )

    if not features.index.equals(target.index):
        raise RuntimeError(
            "Training features and labels have mismatched dates."
        )

    return features, target


# ============================================================
# PIPELINE CONSTRUCTION
# ============================================================


def _create_training_pipeline(
    model_definition: ModelDefinition,
) -> Pipeline:
    """
    Create preprocessing and model-training steps for one estimator.

    Median imputation is fitted only on the current historical training
    sample. This prevents future observations from influencing the missing
    value replacement.

    StandardScaler is used only for models marked as requiring scaling.
    """

    pipeline_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False,
                keep_empty_features=True,
            ),
        )
    ]

    if model_definition.requires_scaling:
        pipeline_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    estimator: BaseEstimator = (
        model_definition.create_fresh_estimator()
    )

    pipeline_steps.append(
        (
            "model",
            estimator,
        )
    )

    return Pipeline(steps=pipeline_steps)


def _fit_single_model(
    model_definition: ModelDefinition,
    features: pd.DataFrame,
    target: pd.Series,
) -> FittedModel:
    """Create and fit one fresh model pipeline."""

    pipeline = _create_training_pipeline(
        model_definition=model_definition,
    )

    try:
        pipeline.fit(
            features,
            target,
        )

    except Exception as error:
        raise RuntimeError(
            f"Training failed for {model_definition.name}: {error}"
        ) from error

    return FittedModel(
        name=model_definition.name,
        pipeline=pipeline,
        weight=model_definition.weight,
        training_rows=len(features),
        positive_class_rate=float(target.mean()),
    )


# ============================================================
# PUBLIC TRAINING FUNCTIONS
# ============================================================


def train_model_ensemble(
    data: pd.DataFrame,
    feature_columns: list[str],
    model: ModelEnsemble,
    settings: Any,
    training_end_date: pd.Timestamp | str | None = None,
) -> FittedEnsemble:
    """
    Train every model in the ensemble using historical information.

    Parameters
    ----------
    data:
        Complete feature dataset.

    feature_columns:
        Exact feature names used by the models.

    model:
        Untrained ModelEnsemble created in models.py.

    settings:
        Quant Machine settings.

    training_end_date:
        Latest date that may be considered when selecting labeled training
        observations. In walk-forward testing, this should be the final date
        available before the prediction date.

    Returns
    -------
    FittedEnsemble
        Trained model collection ready for prediction.
    """

    model.validate()

    features, target = _prepare_training_data(
        data=data,
        feature_columns=feature_columns,
        settings=settings,
        training_end_date=training_end_date,
    )

    fitted_models: dict[str, FittedModel] = {}

    if settings.verbose:
        print(
            f"  Training rows: {len(features):,} "
            f"({features.index.min():%Y-%m-%d} to "
            f"{features.index.max():%Y-%m-%d})"
        )

        print(
            f"  Training labels: "
            f"{target.mean():.2%} UP, "
            f"{1.0 - target.mean():.2%} DOWN"
        )

    for model_name, model_definition in model.models.items():
        if settings.verbose:
            print(f"  Training {model_name}...")

        fitted_model = _fit_single_model(
            model_definition=model_definition,
            features=features,
            target=target,
        )

        fitted_models[model_name] = fitted_model

    if not fitted_models:
        raise RuntimeError(
            "No models were successfully trained."
        )

    return FittedEnsemble(
        models=fitted_models,
        feature_columns=list(feature_columns),
        training_start_date=features.index.min(),
        training_end_date=features.index.max(),
        training_rows=len(features),
        probability_threshold=model.probability_threshold,
        short_probability_threshold=(
            model.short_probability_threshold
        ),
    )


def prepare_prediction_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    prediction_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Select one feature row for prediction.

    This function does not impute or scale the row. Those transformations are
    performed by each fitted training pipeline using values learned only from
    its historical training sample.
    """

    prediction_timestamp = pd.Timestamp(prediction_date)

    if prediction_timestamp not in data.index:
        raise ValueError(
            f"Prediction date {prediction_timestamp:%Y-%m-%d} "
            "does not exist in the dataset."
        )

    _validate_feature_columns(
        data=data,
        feature_columns=feature_columns,
    )

    feature_row = data.loc[
        [prediction_timestamp],
        feature_columns,
    ].copy()

    feature_row = feature_row.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if feature_row.isna().all(axis=1).iloc[0]:
        raise RuntimeError(
            f"Every feature is missing for "
            f"{prediction_timestamp:%Y-%m-%d}."
        )

    return feature_row


def get_training_summary(
    fitted_ensemble: FittedEnsemble,
) -> dict[str, Any]:
    """Return structured information about a fitted ensemble."""

    model_details: dict[str, dict[str, Any]] = {}

    normalized_weights = fitted_ensemble.normalized_weights()

    for model_name, fitted_model in fitted_ensemble.models.items():
        model_details[model_name] = {
            "training_rows": fitted_model.training_rows,
            "positive_class_rate": (
                fitted_model.positive_class_rate
            ),
            "configured_weight": fitted_model.weight,
            "normalized_weight": normalized_weights[model_name],
        }

    return {
        "training_start_date": (
            fitted_ensemble.training_start_date
        ),
        "training_end_date": (
            fitted_ensemble.training_end_date
        ),
        "training_rows": fitted_ensemble.training_rows,
        "feature_count": len(
            fitted_ensemble.feature_columns
        ),
        "models": model_details,
    }