from collections import defaultdict
from dataclasses import replace

from app.detection.base import Anomaly, Severity


class EnsembleDetector:
    """Combines multiple detectors via voting.

    Voting rules:
        1 detector fired  -> skip (not enough evidence)
        2 detectors fired -> WARNING
        3+ detectors fired -> CRITICAL

    Deduplicates by (date, metric) and merges details from all detectors.
    """

    def __init__(self, min_votes_warning: int = 2, min_votes_critical: int = 3):
        self.min_votes_warning = min_votes_warning
        self.min_votes_critical = min_votes_critical

    def combine(self, *anomaly_lists: list[Anomaly]) -> list[Anomaly]:
        """Merge anomaly lists from multiple detectors and apply voting."""
        # Group anomalies by (date, metric)
        groups: dict[tuple[str, str], list[Anomaly]] = defaultdict(list)
        for anomaly_list in anomaly_lists:
            for a in anomaly_list:
                groups[(a.date, a.metric)].append(a)

        results: list[Anomaly] = []

        for (date, metric), detections in sorted(groups.items()):
            n_votes = len(detections)

            if n_votes < self.min_votes_warning:
                continue

            if n_votes >= self.min_votes_critical:
                severity = Severity.CRITICAL
            else:
                severity = Severity.WARNING

            # Use the first detection as base, merge info from all
            base = detections[0]
            detector_names = [a.detector for a in detections]
            all_details = "; ".join(
                f"[{a.detector}] {a.details}" for a in detections
            )

            results.append(
                replace(
                    base,
                    severity=severity,
                    detector=f"Ensemble({n_votes}/{len(anomaly_lists)})",
                    details=(
                        f"Voted by: {', '.join(detector_names)}. "
                        f"Details: {all_details}"
                    ),
                )
            )

        return results
