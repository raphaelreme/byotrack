from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


warnings.warn(
    "`parameters` module is deprecated and should not be used. It may be re-integrated in a future version.",
    DeprecationWarning,
    stacklevel=2,
)


class ParameterRange(ABC):
    """Define a range of validity for parameters."""

    @abstractmethod
    def is_valid(self, parameter: object) -> bool:
        """Check that the parameter's value is in the validity range."""

    @abstractmethod
    def to_str(self) -> str:
        """Returns a string describing the valid range."""


class ParameterBound(ParameterRange):
    """Bounded parameters between low and high values."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    @override
    def is_valid(self, parameter: object) -> bool:
        if isinstance(parameter, (float, int)):
            return self.low <= parameter <= self.high
        return False

    @override
    def to_str(self) -> str:
        return f"[{self.low}, {self.high}]"


class ParameterEnum(ParameterRange):
    """Parameters that can have only a set of predefined value."""

    def __init__(self, values: set) -> None:
        super().__init__()
        self.values = values

    @override
    def is_valid(self, parameter: object) -> bool:
        return parameter in self.values

    @override
    def to_str(self) -> str:
        return str(self.values)


class ParametrizedObjectMixin:
    """Objects with hyper parameters [Deprecated].

    The child classes have to define the `parameters` dict that describes all their parameters
    with a ParameterRange. When the instance tries to set a parameter value, this range is used
    to validate the value.

    Attributes:
        parametrized (dict[str, ParametrizedObjectMixin]): Registered child ParametrizedObjectMixin
            ParametrizedObjectMixin attributes are automatically registered.
        parameters (dict[str, ParameterRange]): Defined range for parameters

    """

    parametrized: dict[str, ParametrizedObjectMixin] = {}  # noqa: RUF012
    parameters: dict[str, ParameterRange] = {}  # noqa: RUF012

    @override
    def __setattr__(self, __name: str, __value: Any) -> None:  # noqa: PYI063
        if isinstance(__value, ParametrizedObjectMixin):
            self.parametrized[__name] = __value

        parameter_range = self.parameters.get(__name)
        if parameter_range is None or parameter_range.is_valid(__value):
            return super().__setattr__(__name, __value)

        raise ValueError(f"Invalid value for parameter {__name}: {__value} not in {parameter_range.to_str()}")
