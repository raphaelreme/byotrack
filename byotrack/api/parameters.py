from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Dict, Set


class ParameterRange(ABC):
    """Define a range of validity for parameters"""

    @abstractmethod
    def is_valid(self, parameter: object) -> bool:
        """Check that the parameter's value is in the validity range"""

    @abstractmethod
    def to_str(self) -> str:
        """Returns a string describing the valid range"""


class ParameterBound(ParameterRange):
    """Bounded parameters between low and high values"""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def is_valid(self, parameter: object) -> bool:
        if isinstance(parameter, (float, int)):
            return self.low <= parameter <= self.high
        return False

    def to_str(self) -> str:
        return f"[{self.low}, {self.high}]"


class ParameterEnum(ParameterRange):
    """Parameters that can have only a set of predefined value"""

    def __init__(self, values: Set) -> None:
        super().__init__()
        self.values = values

    def is_valid(self, parameter: object) -> bool:
        return parameter in self.values

    def to_str(self) -> str:
        return str(self.values)


class ParametrizedObjectMixin:  # pylint: disable=too-few-public-methods
    """Objects with hyper parameters

    The child classes have to define the `parameters` dict that describes all their parameters
    with a ParameterRange. When the instance tries to set a parameter value, this range is used
    to validate the value.

    (See an implementation with SpotDetector)

    Attributes:
        parametrized (Dict[str, ParametrizedObjectMixin]): Registered child ParametrizedObjectMixin
            ParametrizedObjectMixin attributes are automatically registered.
        parameters (Dict[str, ParameterRange]): Defined range for parameters

    """

    parametrized: Dict[str, ParametrizedObjectMixin] = {}
    parameters: Dict[str, ParameterRange] = {}

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, ParametrizedObjectMixin):
            self.parametrized[__name] = __value

        parameter_range = self.parameters.get(__name)
        if parameter_range is None or parameter_range.is_valid(__value):
            return super().__setattr__(__name, __value)

        raise ValueError(f"Invalid value for parameter {__name}: {__value} not in {parameter_range.to_str()}")
