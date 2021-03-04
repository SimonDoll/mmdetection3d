from .metric_result import MetricResult


class NumericMetricResult(MetricResult):
    def __init__(self, result):
        assert isinstance(result, float)
        self._result = result

    def __call__(self):
        return self._result
