from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EventMessage(_message.Message):
    __slots__ = ("event_json", "origin_node_id", "origin_idx")
    EVENT_JSON_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_IDX_FIELD_NUMBER: _ClassVar[int]
    event_json: str
    origin_node_id: str
    origin_idx: int
    def __init__(self, event_json: _Optional[str] = ..., origin_node_id: _Optional[str] = ..., origin_idx: _Optional[int] = ...) -> None: ...

class StateMessage(_message.Message):
    __slots__ = ("state_json",)
    STATE_JSON_FIELD_NUMBER: _ClassVar[int]
    state_json: str
    def __init__(self, state_json: _Optional[str] = ...) -> None: ...

class CommandMessage(_message.Message):
    __slots__ = ("command_json", "origin_node_id")
    COMMAND_JSON_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    command_json: str
    origin_node_id: str
    def __init__(self, command_json: _Optional[str] = ..., origin_node_id: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...
