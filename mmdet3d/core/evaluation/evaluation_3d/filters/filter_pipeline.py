from .filter import Filter


class FilterPipeline(Filter):
    """ applies a chain of filters """

    def __init__(self, filter_objects):
        self._filter_objects = filter_objects

    def apply(
        self, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
    ):
        for f in self._filter_objects:
            (
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                input_data,
            ) = f.apply(
                gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
            )
            assert isinstance(f, Filter)

        return gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data