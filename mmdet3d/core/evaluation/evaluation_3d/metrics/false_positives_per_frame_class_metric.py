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
        return 'FalsePositivesPerFramePerClass'

    def _extract_frame_count(self, matching_results_per_class):
        unique_data_ids = set()
        for m_res in matching_results_per_class:
            unique_data_ids.add(m_res['data_id'])

        return len(unique_data_ids)

    def compute(self, matching_results, data=None):
        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=False,
            reversed_score=self._reversed_score,
        )

        false_positives_per_class = {class_id: 0
                                     for class_id in matching_results.keys()}

        for class_id in matching_results.keys():
            frame_count = self._extract_frame_count(matching_results[class_id])
            if frame_count == 0:
                # no frames for this class -> continue
                continue

            fps = decisions_per_class[class_id]['fps']
            false_positives_per_class[class_id] = float(fps) / frame_count

        return false_positives_per_class

    def evaluate(self, matching_results, data=None):
        false_positives_per_class = self.compute(matching_results, data)

        return self.create_result(false_positives_per_class)
