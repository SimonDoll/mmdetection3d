from abc import ABC, abstractmethod


class MetricResult(ABC):
    """This class serves as interface for metric return types All results are
    callable  and return their result value."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        """returns the result of a metric needs to be overriden by
        baseclass."""
