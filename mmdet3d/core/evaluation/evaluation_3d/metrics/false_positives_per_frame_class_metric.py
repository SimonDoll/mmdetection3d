from .numeric_class_metric import NumericClassMetric


class FalsePositivesPerFrameClassMetric(NumericClassMetric):
    """Calculates mean false positives per frame for each class"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """FalsePositivesPerFramePerClass

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'RecallPerClass'

    def compute(self, matching_results, data=None):
        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=True,
            reversed_score=self._reversed_score,
        )

        false_positives_per_class = {class_id: 0
                                     for class_id in matching_results.keys()}

        for class_id in matching_results.keys():
            frame_count = len(matching_results[class_id])
            if frame_count == 0:
                # no frames for this class -> continue
                continue

            fps = decisions_per_class[class_id]['fps'].sum().item()

            false_positives_per_class[class_id] = float(fps) / frame_count

        return false_positives_per_class

    def evaluate(self, matching_results, data=None):
        false_positives_per_class = self.compute(matching_results, data)

        return self.create_result(false_positives_per_class)
