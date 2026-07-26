"""
Model Definitions
=================

This file defines the machine-learning models used by the Quant Machine.

It does not train the models. Training logic will be placed in training.py.

The initial ensemble contains:

1. Logistic Regression
2. Random Forest
3. Histogram Gradient Boosting

Each model produces a probability that the stock will move upward.
The final ensemble probability is calculated using configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression


# ============================================================
# INDIVIDUAL MODEL CONSTRUCTION
# ============================================================


def create_logistic_regression(settings: Any) -> LogisticRegression:
    """
    Create the Logistic Regression classifier.

    Logistic Regression provides a simpler linear baseline. It helps prevent
    the ensemble from relying entirely on complex tree-based models.

    Feature scaling and missing-value handling will be added in training.py.
    """

    return LogisticRegression(
        C=settings.logistic_regularization,
        max_iter=settings.logistic_max_iterations,
        class_weight="balanced",
        solver="lbfgs",
        random_state=settings.random_state,
    )


def create_random_forest(settings: Any) -> RandomForestClassifier:
    """
    Create the Random Forest classifier.

    Random Forest can capture nonlinear relationships and interactions among
    technical indicators without requiring feature scaling.
    """

    return RandomForestClassifier(
        n_estimators=settings.random_forest_trees,
        max_depth=settings.random_forest_max_depth,
        min_samples_leaf=settings.random_forest_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=settings.random_state,
        n_jobs=settings.n_jobs,
    )


def create_gradient_boosting(
    settings: Any,
) -> HistGradientBoostingClassifier:
    """
    Create the Histogram Gradient Boosting classifier.

    Histogram Gradient Boosting is efficient for medium-sized tabular datasets
    and can learn nonlinear relationships among market features.
    """

    return HistGradientBoostingClassifier(
        learning_rate=settings.gradient_boosting_learning_rate,
        max_iter=settings.gradient_boosting_iterations,
        max_leaf_nodes=settings.gradient_boosting_max_leaf_nodes,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=settings.random_state,
    )


# ============================================================
# MODEL CONTAINER
# ============================================================


@dataclass
class ModelDefinition:
    """
    Stores one model and its ensemble weight.

    Parameters
    ----------
    name:
        Human-readable model name.

    estimator:
        Untrained scikit-learn classifier.

    weight:
        Contribution of the model to the final ensemble probability.

    requires_scaling:
        Whether the model should receive standardized features.
    """

    name: str
    estimator: BaseEstimator
    weight: float
    requires_scaling: bool = False

    def create_fresh_estimator(self) -> BaseEstimator:
        """
        Return an unfitted copy of the estimator.

        A fresh estimator is required for every walk-forward backtest step.
        This prevents models from accidentally retaining information from
        later training periods.
        """

        return clone(self.estimator)


@dataclass
class ModelEnsemble:
    """
    Container for all model definitions in the ensemble.

    This object stores model templates rather than fitted models. The templates
    are cloned and trained inside training.py.
    """

    models: dict[str, ModelDefinition]
    probability_threshold: float
    short_probability_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_names(self) -> list[str]:
        """Return all model names in display order."""

        return list(self.models.keys())

    @property
    def total_weight(self) -> float:
        """Return the sum of all ensemble weights."""

        return sum(
            model_definition.weight
            for model_definition in self.models.values()
        )

    def get_definition(self, model_name: str) -> ModelDefinition:
        """
        Return one model definition by name.

        Raises
        ------
        KeyError
            If the requested model does not exist.
        """

        if model_name not in self.models:
            available_models = ", ".join(self.model_names)

            raise KeyError(
                f"Unknown model {model_name!r}. "
                f"Available models: {available_models}"
            )

        return self.models[model_name]

    def normalized_weights(self) -> dict[str, float]:
        """
        Return model weights normalized to add up to 1.0.

        Normalization protects the ensemble if the configured weights differ
        slightly from exactly 1.0.
        """

        total = self.total_weight

        if total <= 0:
            raise ValueError(
                "The combined ensemble weight must be greater than zero."
            )

        return {
            model_name: definition.weight / total
            for model_name, definition in self.models.items()
        }

    def validate(self) -> None:
        """Validate the ensemble configuration."""

        if not self.models:
            raise ValueError(
                "The model ensemble must contain at least one model."
            )

        for model_name, definition in self.models.items():
            if not model_name.strip():
                raise ValueError(
                    "Every model must have a nonempty name."
                )

            if definition.weight < 0:
                raise ValueError(
                    f"Model {model_name!r} has a negative weight."
                )

            if not hasattr(definition.estimator, "fit"):
                raise TypeError(
                    f"Model {model_name!r} does not implement fit()."
                )

            if not hasattr(definition.estimator, "predict_proba"):
                raise TypeError(
                    f"Model {model_name!r} does not implement "
                    "predict_proba()."
                )

        if self.total_weight <= 0:
            raise ValueError(
                "At least one model must have a positive ensemble weight."
            )

        if not 0.50 <= self.probability_threshold <= 1.00:
            raise ValueError(
                "The long probability threshold must be between "
                "0.50 and 1.00."
            )

        if not 0.00 <= self.short_probability_threshold <= 0.50:
            raise ValueError(
                "The short probability threshold must be between "
                "0.00 and 0.50."
            )

        if (
            self.short_probability_threshold
            > self.probability_threshold
        ):
            raise ValueError(
                "The short probability threshold cannot exceed "
                "the long probability threshold."
            )

    def describe(self) -> list[dict[str, Any]]:
        """
        Return a structured description of the ensemble.

        This can later be printed, logged, or exported.
        """

        normalized = self.normalized_weights()

        descriptions: list[dict[str, Any]] = []

        for model_name, definition in self.models.items():
            descriptions.append(
                {
                    "name": model_name,
                    "estimator": type(definition.estimator).__name__,
                    "configured_weight": definition.weight,
                    "normalized_weight": normalized[model_name],
                    "requires_scaling": definition.requires_scaling,
                }
            )

        return descriptions


# ============================================================
# ENSEMBLE CREATION
# ============================================================


def create_model_ensemble(settings: Any) -> ModelEnsemble:
    """
    Create the complete untrained model ensemble.

    Parameters
    ----------
    settings:
        QuantSettings instance imported from user_inputs.py.

    Returns
    -------
    ModelEnsemble
        Validated ensemble containing three untrained model definitions.
    """

    logistic_model = ModelDefinition(
        name="Logistic Regression",
        estimator=create_logistic_regression(settings),
        weight=settings.logistic_weight,
        requires_scaling=True,
    )

    random_forest_model = ModelDefinition(
        name="Random Forest",
        estimator=create_random_forest(settings),
        weight=settings.random_forest_weight,
        requires_scaling=False,
    )

    gradient_boosting_model = ModelDefinition(
        name="Gradient Boosting",
        estimator=create_gradient_boosting(settings),
        weight=settings.gradient_boosting_weight,
        requires_scaling=False,
    )

    models = {
        logistic_model.name: logistic_model,
        random_forest_model.name: random_forest_model,
        gradient_boosting_model.name: gradient_boosting_model,
    }

    ensemble = ModelEnsemble(
        models=models,
        probability_threshold=(
            settings.long_probability_threshold
        ),
        short_probability_threshold=(
            settings.short_probability_threshold
        ),
        metadata={
            "random_state": settings.random_state,
            "target_horizon": settings.target_horizon,
        },
    )

    ensemble.validate()

    return ensemble