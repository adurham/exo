"""Reactive variable utilities.

This module provides a reactive variable type that notifies callbacks
when values change, useful for implementing reactive programming patterns.
"""

from typing import Protocol


class OnChange[T](Protocol):
    """Protocol for change notification callbacks.

    Callbacks of this type are invoked when a reactive variable's value changes.
    """

    def __call__(self, old_value: T, new_value: T) -> None:
        """Called when a value changes.

        Args:
            old_value: Previous value.
            new_value: New value.
        """
        ...


class Reactive[T]:
    """Reactive variable that notifies on value changes.

    Wraps a value and calls a callback whenever the value is changed.
    Only notifies if the new value differs from the old value.

    Attributes:
        _value: Current value of the reactive variable.
        _on_change: Callback to invoke when value changes.
    """

    def __init__(self, initial_value: T, on_change: OnChange[T]):
        """Initialize reactive variable.

        Args:
            initial_value: Initial value.
            on_change: Callback to invoke when value changes.
        """
        self._value = initial_value
        self._on_change = on_change

    @property
    def value(self) -> T:
        """Get current value.

        Returns:
            Current value.
        """
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        """Set new value and notify if changed.

        Args:
            new_value: New value to set.
        """
        old_value = self._value
        self._value = new_value

        if old_value == new_value:
            return

        self._on_change(old_value=old_value, new_value=new_value)
