from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Data paths
    raw_data_path: str = "data/raw/online_retail_II.csv"
    processed_data_path: str = "data/processed/daily_metrics.csv"

    # Z-Score detector
    zscore_window: int = 30
    zscore_warning_threshold: float = 2.0
    zscore_critical_threshold: float = 3.0

    # IQR detector
    iqr_window: int = 30
    iqr_warning_multiplier: float = 1.5
    iqr_critical_multiplier: float = 3.0

    # CUSUM detector
    cusum_drift_factor: float = 0.5
    cusum_warning_factor: float = 4.0
    cusum_critical_factor: float = 6.0

    # Prophet detector
    prophet_interval_warning: float = 0.95
    prophet_interval_critical: float = 0.99

    # Isolation Forest
    iforest_contamination: float = 0.05
    iforest_warning_score: float = -0.15
    iforest_critical_score: float = -0.25

    # Ensemble
    ensemble_warning_votes: int = 2
    ensemble_critical_votes: int = 3

    # Telegram (optional)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
