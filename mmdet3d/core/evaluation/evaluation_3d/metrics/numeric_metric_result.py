from .metric_result import MetricResult


class NumericMetricResult(MetricResult):
    """Interface for metrics that produce a numeric value for the entire eval
    e.g. MeanAveragePrecision."""

    def __init__(self, result):
        assert isinstance(result, float)
        self._result = result

    def __call__(self):
        return self._result

    def __float__(self):
        return self._result
