from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Path(_message.Message):
    __slots__ = ["poses"]
    POSES_FIELD_NUMBER: _ClassVar[int]
    poses: _containers.RepeatedCompositeFieldContainer[Pose]
    def __init__(self, poses: _Optional[_Iterable[_Union[Pose, _Mapping]]] = ...) -> None: ...

class Pose(_message.Message):
    __slots__ = ["x", "y", "yaw"]
    X_FIELD_NUMBER: _ClassVar[int]
    YAW_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    yaw: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., yaw: _Optional[float] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ["drive_acc", "drive_velocity", "pose", "steer_angle", "steer_velocity"]
    DRIVE_ACC_FIELD_NUMBER: _ClassVar[int]
    DRIVE_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    STEER_ANGLE_FIELD_NUMBER: _ClassVar[int]
    STEER_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    drive_acc: float
    drive_velocity: float
    pose: Pose
    steer_angle: float
    steer_velocity: float
    def __init__(self, drive_velocity: _Optional[float] = ..., drive_acc: _Optional[float] = ..., steer_angle: _Optional[float] = ..., steer_velocity: _Optional[float] = ..., pose: _Optional[_Union[Pose, _Mapping]] = ...) -> None: ...
