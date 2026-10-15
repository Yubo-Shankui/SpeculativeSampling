"""Small profiled EMA; cold/error observations stay out of steady buckets."""
from dataclasses import dataclass
import math


@dataclass
class ProductionEstimate:
    initial_seconds: float
    margin_seconds: float
    alpha: float = 0.2
    count: int = 0

    def __post_init__(self):
        if not (math.isfinite(self.initial_seconds) and self.initial_seconds > 0
                and math.isfinite(self.margin_seconds) and self.margin_seconds >= 0
                and 0 <= self.alpha <= 1):
            raise ValueError("invalid calibrated estimator")
        self.value = self.initial_seconds
        self.excluded = []

    def observe(self, seconds, *, cold=False, failed=False):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("duration must be finite and non-negative")
        if cold or failed:
            self.excluded.append({"seconds": seconds, "cold": cold, "failed": failed})
            return
        if self.alpha == 0:
            return
        self.value = self.alpha * seconds + (1 - self.alpha) * self.value
        self.count += 1

    @property
    def lead_seconds(self):
        return self.value + self.margin_seconds
