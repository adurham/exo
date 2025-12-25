import pytest

from exo.utils.reactive import OnChange, Reactive


def test_reactive_initial_value():
    """Test that Reactive stores the initial value."""
    reactive = Reactive(initial_value=42, on_change=lambda old, new: None)
    assert reactive.value == 42


def test_reactive_notifies_on_change():
    """Test that Reactive calls on_change when value changes."""
    changes = []

    def on_change(old_value: int, new_value: int) -> None:
        changes.append((old_value, new_value))

    reactive = Reactive(initial_value=10, on_change=on_change)
    reactive.value = 20
    assert changes == [(10, 20)]


def test_reactive_notifies_multiple_changes():
    """Test that Reactive calls on_change for each change."""
    changes = []

    def on_change(old_value: int, new_value: int) -> None:
        changes.append((old_value, new_value))

    reactive = Reactive(initial_value=0, on_change=on_change)
    reactive.value = 10
    reactive.value = 20
    reactive.value = 30
    assert changes == [(0, 10), (10, 20), (20, 30)]


def test_reactive_no_notification_when_unchanged():
    """Test that Reactive does not notify when value doesn't change."""
    changes = []

    def on_change(old_value: int, new_value: int) -> None:
        changes.append((old_value, new_value))

    reactive = Reactive(initial_value=42, on_change=on_change)
    reactive.value = 42  # Same value
    assert changes == []  # No notification


def test_reactive_with_strings():
    """Test Reactive with string values."""
    changes = []

    def on_change(old_value: str, new_value: str) -> None:
        changes.append((old_value, new_value))

    reactive = Reactive(initial_value="hello", on_change=on_change)
    reactive.value = "world"
    assert changes == [("hello", "world")]


def test_reactive_with_lists():
    """Test Reactive with list values."""
    changes = []

    def on_change(old_value: list, new_value: list) -> None:
        changes.append((old_value, new_value))

    reactive = Reactive(initial_value=[1, 2, 3], on_change=on_change)
    reactive.value = [4, 5, 6]
    assert len(changes) == 1
    assert changes[0][0] == [1, 2, 3]
    assert changes[0][1] == [4, 5, 6]


def test_reactive_protocol():
    """Test that OnChange protocol is properly defined."""
    # This test verifies the protocol exists and can be used for type checking
    def callback(old: int, new: int) -> None:
        pass

    # Should not raise any errors
    reactive = Reactive(initial_value=0, on_change=callback)
    assert reactive.value == 0

