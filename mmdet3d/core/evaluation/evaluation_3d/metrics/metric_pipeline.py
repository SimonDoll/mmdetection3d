import warnings
from beautifultable import BeautifulTable

from .base_metric import Basemetric
from .numeric_class_metric_result import NumericClassMetricResult
from .numeric_metric_result import NumericMetricResult


class MetricPipeline:

    def __init__(self, metric_objects):
        self._metric_objects = metric_objects

    def __str__(self):
        return 'MetricPipeline'

    def evaluate(self, matching_results, data=None):
        # TODO split in class and numeric metrics
        results = {}
        for metric in self._metric_objects:
            assert isinstance(metric, Basemetric)

            metric_result = metric.evaluate(matching_results, data)
            results[str(metric)] = metric_result

        return results

    @staticmethod
    def print_results(metric_results):

        # name x value
        results_numeric = {}

        # {metric :{x classname : value}}
        class_results_numeric = {}

        for metric_name, metric_return in metric_results.items():

            if isinstance(metric_return, NumericMetricResult):
                # this value is a number
                results_numeric[metric_name] = float(metric_return())
            elif isinstance(metric_return, NumericClassMetricResult):
                # this return val is a dict  classes x numeric value
                for class_name, value in metric_return().items():
                    if not metric_name in class_results_numeric:
                        # this class was  not added before -> add
                        class_results_numeric[metric_name] = {}

                    # sanity check that we do not override something
                    assert class_name not in class_results_numeric[metric_name]

                    class_results_numeric[metric_name][class_name] = value
            else:
                warnings.warn(
                    'The following metrics result is of an unprintable type and therefore ignored: {}'
                    .format(metric_name))
        # print the numeric results first
        numeric_table = BeautifulTable()
        numeric_table.header = ['Metric', 'Value']
        for metric_name, numeric_result in results_numeric.items():
            numeric_table.rows.append([metric_name, numeric_result])

        print('=' * 40)
        print('Evaluation results')
        print('=' * 40)
        print(numeric_table)

        numeric_per_class_table = BeautifulTable()
        header = None
        for metric_name in class_results_numeric:
            # sort the classes to get the same order for all rows
            classes = sorted(class_results_numeric[metric_name])
            if not header:
                # add the class names to header
                header = classes

            # assert that the same classes are used for all metrics
            assert header == classes

            row = [metric_name]
            for class_id in class_results_numeric[metric_name]:
                value = class_results_numeric[metric_name][class_id]()
                row.append(value)

            numeric_per_class_table.rows.append(row)

        header = list(map(str, header))
        numeric_per_class_table.header = ['Metric'] + header

        print(numeric_per_class_table)
