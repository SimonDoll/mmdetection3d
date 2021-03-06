from .numeric_metric import NumericMetric
from .false_positives_per_frame_class_metric import FalsePositivesPerFrameClassMetric


class FalsePositivesPerFrame(NumericMetric):
    """Calculates mean false positives per frame (avg over classes)"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """FalsePositivesPerFrame

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)
        self._fp_pf_per_class_metric = FalsePositivesPerFrameClassMetric(
            similarity_threshold, reversed_score)

    def __str__(self):
        return 'FalsePositivesPerFrame'

    def evaluate(self, matching_results, data=None):
        fp_pf__per_class = self._fp_pf_per_class_metric.compute(
            matching_results, data)

        fps_per_frame = 0.0
        classes = 0
        for c, fps_per_frame_c in fp_pf__per_class.items():
            fps_per_frame += fps_per_frame_c
            classes += 1
        fps_per_frame /= classes

        return self.create_result(fps_per_frame)
