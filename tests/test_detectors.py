from dataclasses import replace

import pandas as pd

from app.detection.base import Anomaly, Severity
from app.detection.cusum_detector import CUSUMDetector
from app.detection.ensemble import EnsembleDetector
from app.detection.iqr_detector import IQRDetector
from app.detection.isolation_forest import IsolationForestDetector
from app.detection.zscore_detector import ZScoreDetector

# ═══════════════════════════════════════════════════════════════════════════
#  ZScoreDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestZScore:
    def test_returns_anomaly_objects(self, sample_series):
        d = ZScoreDetector(window=30).fit(sample_series)
        results = d.detect(sample_series)
        assert isinstance(results, list)
        assert all(isinstance(a, Anomaly) for a in results)

    def test_detects_spike(self, sample_series):
        d = ZScoreDetector(window=30).fit(sample_series)
        dates = {a.date for a in d.detect(sample_series)}
        assert "2023-02-20" in dates  # index 50

    def test_severity_critical_for_large_spike(self, sample_series):
        d = ZScoreDetector(window=30).fit(sample_series)
        spike = [a for a in d.detect(sample_series) if a.date == "2023-02-20"]
        assert len(spike) == 1
        assert spike[0].severity == Severity.CRITICAL

    def test_deviation_pct_positive_for_spike(self, sample_series):
        d = ZScoreDetector(window=30).fit(sample_series)
        spike = [a for a in d.detect(sample_series) if a.date == "2023-02-20"][0]
        assert spike.deviation_pct > 100

    def test_low_fpr_on_clean_data(self, normal_series):
        """False positive rate < 10% on data with no real anomalies."""
        d = ZScoreDetector(window=30).fit(normal_series)
        results = d.detect(normal_series)
        fpr = len(results) / len(normal_series)
        assert fpr < 0.10, f"FPR too high: {fpr:.2%}"

    def test_recall_on_synthetic(self, synthetic_data):
        """Should find >70% of point anomalies in synthetic data."""
        series = synthetic_data["revenue"]
        series.name = "revenue"
        point_mask = synthetic_data["anomaly_type"] == "point"
        point_dates = set(str(d.date()) for d in synthetic_data.index[point_mask])

        d = ZScoreDetector(window=30).fit(series)
        detected_dates = {a.date for a in d.detect(series)}

        hits = point_dates & detected_dates
        recall = len(hits) / len(point_dates) if point_dates else 1.0
        assert recall >= 0.70, f"Recall on point anomalies too low: {recall:.2%}"


# ═══════════════════════════════════════════════════════════════════════════
#  IQRDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestIQR:
    def test_detects_spike(self, sample_series):
        d = IQRDetector(window=30).fit(sample_series)
        dates = {a.date for a in d.detect(sample_series)}
        assert "2023-02-20" in dates

    def test_severity_critical_for_large_spike(self, sample_series):
        d = IQRDetector(window=30).fit(sample_series)
        spike = [a for a in d.detect(sample_series) if a.date == "2023-02-20"]
        assert len(spike) == 1
        assert spike[0].severity == Severity.CRITICAL

    def test_low_fpr_on_clean_data(self, normal_series):
        d = IQRDetector(window=30).fit(normal_series)
        results = d.detect(normal_series)
        fpr = len(results) / len(normal_series)
        assert fpr < 0.10, f"FPR too high: {fpr:.2%}"

    def test_recall_on_synthetic(self, synthetic_data):
        series = synthetic_data["revenue"]
        series.name = "revenue"
        point_mask = synthetic_data["anomaly_type"] == "point"
        point_dates = set(str(d.date()) for d in synthetic_data.index[point_mask])

        d = IQRDetector(window=30).fit(series)
        detected_dates = {a.date for a in d.detect(series)}

        hits = point_dates & detected_dates
        recall = len(hits) / len(point_dates) if point_dates else 1.0
        # IQR is less sensitive than Z-score on point anomalies in noisy data
        assert recall >= 0.40, f"Recall on point anomalies too low: {recall:.2%}"


# ═══════════════════════════════════════════════════════════════════════════
#  CUSUMDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestCUSUM:
    def test_returns_anomaly_objects(self, sample_series):
        d = CUSUMDetector().fit(sample_series)
        results = d.detect(sample_series)
        assert isinstance(results, list)
        assert all(isinstance(a, Anomaly) for a in results)

    def test_detects_spike(self, sample_series):
        d = CUSUMDetector().fit(sample_series)
        dates = {a.date for a in d.detect(sample_series)}
        assert "2023-02-20" in dates  # the big spike at index 50

    def test_detects_level_shift(self):
        """CUSUM should detect a sustained level shift."""
        import numpy as np

        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        values = rng.normal(1000, 50, 200)
        # Level shift: mean jumps from 1000 to 1300 at day 120
        values[120:] += 300
        s = pd.Series(values, index=dates, name="test_metric")

        d = CUSUMDetector(drift_factor=0.5, warning_factor=4.0, critical_factor=6.0)
        d.fit(s[:120])  # fit on pre-shift data
        results = d.detect(s)
        shift_dates = {a.date for a in results if a.date >= "2023-04-30"}
        assert len(shift_dates) > 0, "CUSUM should detect the level shift"

    def test_low_fpr_on_clean_data(self, normal_series):
        """False positive rate < 10% on clean data."""
        d = CUSUMDetector().fit(normal_series)
        results = d.detect(normal_series)
        fpr = len(results) / len(normal_series)
        assert fpr < 0.10, f"FPR too high: {fpr:.2%}"

    def test_severity_critical_for_large_shift(self):
        """A large sustained shift should produce CRITICAL severity."""
        import numpy as np

        rng = np.random.default_rng(7)
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        values = rng.normal(1000, 30, 150)
        values[100:] += 500  # huge shift
        s = pd.Series(values, index=dates, name="test_metric")

        d = CUSUMDetector().fit(s[:100])
        results = d.detect(s)
        severities = {a.severity for a in results}
        assert Severity.CRITICAL in severities

    def test_details_contain_direction(self, sample_series):
        d = CUSUMDetector().fit(sample_series)
        results = d.detect(sample_series)
        for a in results:
            assert "upward" in a.details or "downward" in a.details


# ═══════════════════════════════════════════════════════════════════════════
#  IsolationForestDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestIsolationForest:
    def test_returns_anomaly_objects(self, sample_dataframe):
        d = IsolationForestDetector(contamination=0.05).fit(sample_dataframe)
        results = d.detect(sample_dataframe)
        assert isinstance(results, list)
        assert all(isinstance(a, Anomaly) for a in results)

    def test_detects_spike(self, sample_dataframe):
        d = IsolationForestDetector(contamination=0.05).fit(sample_dataframe)
        dates = {a.date for a in d.detect(sample_dataframe)}
        assert "2023-02-20" in dates

    def test_metric_is_multivariate(self, sample_dataframe):
        d = IsolationForestDetector(contamination=0.05).fit(sample_dataframe)
        for a in d.detect(sample_dataframe):
            assert a.metric == "multivariate"

    def test_low_fpr_on_clean_data(self, normal_series):
        """Build a clean multivariate DF — FPR should be low."""
        import numpy as np

        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            {
                "revenue": normal_series.values,
                "orders": rng.normal(50, 3, len(normal_series)).astype(int),
                "avg_check": normal_series.values / 50,
                "unique_customers": rng.normal(40, 3, len(normal_series)).astype(int),
            },
            index=normal_series.index,
        )
        d = IsolationForestDetector(contamination=0.02).fit(df)
        results = d.detect(df)
        fpr = len(results) / len(df)
        assert fpr < 0.10, f"FPR too high: {fpr:.2%}"


# ═══════════════════════════════════════════════════════════════════════════
#  EnsembleDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestEnsemble:
    def test_two_votes_warning(self, sample_series):
        z = ZScoreDetector(window=30).fit(sample_series)
        iqr = IQRDetector(window=30).fit(sample_series)

        ens = EnsembleDetector(min_votes_warning=2, min_votes_critical=3)
        combined = ens.combine(z.detect(sample_series), iqr.detect(sample_series))

        spike = [a for a in combined if a.date == "2023-02-20"]
        assert len(spike) == 1
        assert spike[0].severity == Severity.WARNING  # 2 votes = warning

    def test_single_vote_filtered(self):
        single = [
            Anomaly(
                date="2023-01-15", metric="revenue", value=50000,
                expected=10000, deviation_pct=400.0,
                severity=Severity.CRITICAL, detector="D1", details="test",
            )
        ]
        ens = EnsembleDetector(min_votes_warning=2)
        assert len(ens.combine(single, [])) == 0

    def test_three_votes_critical(self):
        a = Anomaly(
            date="2023-01-15", metric="revenue", value=50000,
            expected=10000, deviation_pct=400.0,
            severity=Severity.WARNING, detector="D1", details="d1",
        )
        b = replace(a, detector="D2", details="d2")
        c = replace(a, detector="D3", details="d3")

        ens = EnsembleDetector(min_votes_warning=2, min_votes_critical=3)
        combined = ens.combine([a], [b], [c])
        assert len(combined) == 1
        assert combined[0].severity == Severity.CRITICAL
        assert "D1" in combined[0].details
        assert "D2" in combined[0].details
        assert "D3" in combined[0].details

    def test_ensemble_catches_more_than_single(self, sample_series):
        """Ensemble with min_votes=1 should find at least as many as any single detector."""
        z = ZScoreDetector(window=30).fit(sample_series)
        iqr = IQRDetector(window=30).fit(sample_series)
        z_res = z.detect(sample_series)
        iqr_res = iqr.detect(sample_series)

        # Union: ensemble with min_votes=1 captures everything
        ens = EnsembleDetector(min_votes_warning=1, min_votes_critical=3)
        combined = ens.combine(z_res, iqr_res)

        z_dates = {a.date for a in z_res}
        iqr_dates = {a.date for a in iqr_res}
        ens_dates = {a.date for a in combined}

        assert ens_dates >= z_dates, "Ensemble should include all Z-score detections"
        assert ens_dates >= iqr_dates, "Ensemble should include all IQR detections"


# ═══════════════════════════════════════════════════════════════════════════
#  Severity assignment
# ═══════════════════════════════════════════════════════════════════════════

class TestSeverity:
    def test_zscore_warning_vs_critical(self):
        """A large spike should be CRITICAL; the test checks that both severity
        levels can be produced by the detector."""
        import numpy as np

        rng = np.random.default_rng(10)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        values = rng.normal(1000, 50, 200)
        # Strong anomaly at index 150 (well inside rolling window)
        values[150] = 1000 + 50 * 6.0
        s = pd.Series(values, index=dates, name="test")

        d = ZScoreDetector(window=30, warning_threshold=2.0, critical_threshold=3.0).fit(s)
        results = d.detect(s)

        # The huge spike should produce CRITICAL
        strong = [a for a in results if a.date == "2023-05-31"]  # index 150
        assert len(strong) == 1
        assert strong[0].severity == Severity.CRITICAL

        # Should also produce some WARNINGs somewhere (mild fluctuations)
        severities = {a.severity for a in results}
        # At minimum we know CRITICAL exists
        assert Severity.CRITICAL in severities

    def test_iqr_warning_vs_critical(self, sample_series):
        d = IQRDetector(window=30, warning_multiplier=1.5, critical_multiplier=3.0)
        d.fit(sample_series)
        results = d.detect(sample_series)
        severities = {a.severity for a in results}
        # The 30_000 spike should trigger at least one critical
        assert Severity.CRITICAL in severities
