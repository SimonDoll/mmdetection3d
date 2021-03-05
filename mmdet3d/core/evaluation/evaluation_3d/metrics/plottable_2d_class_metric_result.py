from .metric_result import MetricResult
from .plottable_2d_metric_result import Plottable2dMetricResult


class Plottable2dClassMetricResult(MetricResult):
    """Interface for metric results that should be plotted in 2d on a per class
    basis."""

    def __init__(self, result):
        assert isinstance(result, dict)

        for class_name, val in result.items():
            assert isinstance(val, Plottable2dMetricResult)
        self._result = result

    def __call__(self):
        return self._result
