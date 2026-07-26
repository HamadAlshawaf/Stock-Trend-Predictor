"""
User Inputs
===========

This is the main configuration file for the Quant Machine.

Most of the time, this should be the only file you edit before running
the program.

You can control:

- Which stock to analyze
- Historical or live prediction mode
- Training start date
- Historical cutoff date
- Number of prediction days
- Model thresholds
- Backtest assumptions
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSettings:
    """
    Stores all user-editable settings for the Quant Machine.

    The dataclass is frozen so settings cannot accidentally change while
    the program is running.
    """

    # ============================================================
    # PRIMARY STOCK SETTINGS
    # ============================================================

    # Yahoo Finance ticker symbol.
    #
    # Examples:
    # "ELDN"
    # "AAPL"
    # "NVDA"
    # "TSLA"
    ticker: str = "GOOGL"

    # Earliest date used to download historical market data.
    #
    # Format:
    # YYYY-MM-DD
    start_date: str = "2010-01-01"

    # ============================================================
    # PREDICTION MODE
    # ============================================================

    # Available modes:
    #
    # "historical"
    # Train through the cutoff date and evaluate the next
    # prediction_days completed trading days.
    #
    # "live"
    # Train through the latest completed trading day and predict
    # the next trading day.
    mode: str = "live"

    # Last date available to the model before historical testing begins.
    #
    # This is only used when mode = "historical".
    #
    # Example:
    # cutoff_date = "2026-07-15"
    #
    # The first prediction will be for the next trading day after
    # July 15, 2026.
    cutoff_date: str = "2026-06-26"

    # Number of completed trading days to predict and evaluate after
    # the historical cutoff date.
    #
    # This is used only in historical mode.
    #
    # Example:
    # cutoff_date = "2026-07-15"
    # prediction_days = 5
    #
    # The model evaluates:
    # July 16
    # July 17
    # July 20
    # July 21
    # July 22
    prediction_days: int = 10

    # ============================================================
    # PREDICTION TARGET
    # ============================================================

    # Number of trading days ahead the model tries to predict.
    #
    # 1 means:
    # Predict whether the next trading day's close will be higher
    # than today's close.
    #
    # Later, this can be changed to 3, 5, or 10 for longer horizons.
    target_horizon: int = 1

    # Minimum future return required for the target to be labeled UP.
    #
    # 0.00 means any positive return is UP.
    #
    # 0.01 means the stock must rise more than 1%.
    #
    # Keep this at 0.00 for the first version.
    target_return_threshold: float = 0.00

    # ============================================================
    # EXTERNAL MARKET CONTEXT
    # ============================================================

    # Broad US market benchmark.
    market_ticker: str = "^GSPC"

    # Sector benchmark.
    #
    # XBI is appropriate for biotechnology stocks such as ELDN.
    sector_ticker: str = "XBI"

    # Nasdaq Composite.
    nasdaq_ticker: str = "^IXIC"

    # CBOE Volatility Index.
    vix_ticker: str = "^VIX"

    # Russell 2000 small-cap index.
    small_cap_ticker: str = "^RUT"

    # ============================================================
    # MODEL SETTINGS
    # ============================================================

    # Reproducibility seed.
    #
    # Keeping this constant makes model results more consistent
    # between repeated runs.
    random_state: int = 42

    # Minimum number of completed historical rows required before
    # a prediction can be made.
    minimum_training_rows: int = 500

    # Probability required for the model to enter a long trade.
    #
    # 0.55 means the estimated probability of an upward move must
    # be at least 55%.
    long_probability_threshold: float = 0.51

    # Probability at or below which the model labels the signal DOWN.
    #
    # The first version will not automatically short the stock.
    short_probability_threshold: float = 0.49

    # Retrain the models before every historical prediction.
    #
    # This should remain True for realistic walk-forward testing.
    retrain_every_step: bool = True

    # Number of CPU cores available to supported models.
    #
    # -1 means use every available CPU core.
    n_jobs: int = -1

    # ============================================================
    # MODEL-SPECIFIC SETTINGS
    # ============================================================

    # Logistic Regression
    logistic_max_iterations: int = 2_000
    logistic_regularization: float = 0.50

    # Random Forest
    random_forest_trees: int = 500
    random_forest_max_depth: int = 8
    random_forest_min_samples_leaf: int = 10

    # Gradient Boosting
    gradient_boosting_iterations: int = 250
    gradient_boosting_learning_rate: float = 0.04
    gradient_boosting_max_leaf_nodes: int = 15

    # Ensemble model weights.
    #
    # These should add up to 1.00.
    logistic_weight: float = 0.20
    random_forest_weight: float = 0.60
    gradient_boosting_weight: float = 0.20

    # ============================================================
    # BACKTEST SETTINGS
    # ============================================================

    # Estimated total cost per completed trade.
    #
    # This approximates:
    # - Bid-ask spread
    # - Slippage
    # - Commissions
    #
    # 0.001 = 0.10%
    transaction_cost: float = 0.001

    # Starting account value for strategy reports.
    initial_capital: float = 10_000.00

    # Fraction of available capital allocated to each trade.
    #
    # 1.00 = 100%
    # 0.50 = 50%
    # 0.25 = 25%
    position_size: float = 1.00

    # Allow long trades.
    allow_long_positions: bool = True

    # Allow short trades.
    #
    # Keep False for the first version.
    allow_short_positions: bool = False

    # ============================================================
    # DATA SETTINGS
    # ============================================================

    # Automatically adjust historical prices for stock splits
    # and dividends when downloading data.
    auto_adjust_prices: bool = False

    # Number of extra calendar days downloaded after a historical
    # cutoff date.
    #
    # This allows enough future trading days for backtesting.
    future_download_buffer_days: int = 60

    # Forward-fill missing external-market values when the target stock
    # traded but a context series has a missing row.
    forward_fill_market_context: bool = True

    # ============================================================
    # OUTPUT SETTINGS
    # ============================================================

    # Save detailed historical predictions as a CSV file.
    save_backtest_csv: bool = True

    # File name used for historical backtest results.
    backtest_output_file: str = "backtest_results.csv"

    # Display probabilities from each individual model.
    show_individual_model_predictions: bool = True

    # Show additional diagnostic details in the terminal.
    verbose: bool = True

    # Re-raise unexpected errors and show the complete traceback.
    #
    # Set this to True while debugging.
    debug: bool = False


SETTINGS = QuantSettings()