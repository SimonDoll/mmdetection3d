from .metric_result import MetricResult


class Plottable2dMetricResult(MetricResult):
    def __init__(self, x, y, x_name, y_name):
        assert list(x)
        assert list(y)
        self._result = {
            "x": {"name": x_name, "vals": list(x)},
            "y": {"name": y_name, "vals": list(y)},
        }

    def __call__(self):
        return self._result
