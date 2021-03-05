from .metric_result import MetricResult
from .numeric_metric_result import NumericMetricResult


class NumericClassMetricResult(MetricResult):
    """Interface for metrics that produce a numeric value per class.

    Result is a dict {class_id : value}
    """

    def __init__(self, result):
        assert isinstance(result, dict)

        for class_name, val in result.items():
            assert isinstance(val, NumericMetricResult)
        self._result = result

    def __call__(self):
        return self._result
