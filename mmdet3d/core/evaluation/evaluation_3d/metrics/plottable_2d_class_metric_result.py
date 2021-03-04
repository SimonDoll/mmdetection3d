from .metric_result import MetricResult
from .plottable_2d_metric_result import Plottable2dMetricResult


class Plottable2dClassMetricResult(MetricResult):
    def __init__(self, result):
        assert isinstance(result, dict)

        for class_name, val in result.items():
            assert isinstance(val, Plottable2dMetricResult)
        self._result = result

    def __call__(self):
        return self._result
