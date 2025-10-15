"""
Model classes for actuators and sensors
Converted from C++ models.hpp and models.cpp
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from .utils import *


@dataclass
class SpecialParam:
    """Special parameter structure"""
    name: str
    value: str


class AcuratorModel(ABC):
    """Base class for actuator models"""
    
    def __init__(self):
        self.start_p: int = 0
        self.length: int = 0
        self.end_p: int = 0
        self.is_need_parsed: bool = False
        self.buf: Optional[bytearray] = None
    
    def get_length(self) -> int:
        """Get length of the model"""
        return self.length
    
    def get_end_p(self) -> int:
        """Get end position"""
        return self.end_p
    
    def is_need_parsed(self) -> bool:
        """Check if model needs parsing"""
        return self.is_need_parsed
    
    def print_info(self):
        """Print model information (for debugging)"""
        pass
    
    @abstractmethod
    def solve_value(self, output: List[float]):
        """Solve value based on input"""
        pass
    
    @abstractmethod
    def set_value(self, value: bytes) -> int:
        """Set value from bytes"""
        pass
    
    @abstractmethod
    def get_value(self) -> bytes:
        """Get value as bytes"""
        pass
    
    @abstractmethod
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure the model"""
        pass


class CommonParsedValueModel(AcuratorModel):
    """Common parsed value model with signed/unsigned and 16/32 bit support"""
    
    def __init__(self):
        super().__init__()
        self.is_signed: bool = False
        self.is32bits: bool = False
        self.val1: int = 0  # signed value
        self.val2: int = 0  # unsigned value
    
    def set_value(self, value: bytes) -> int:
        """Set value from bytes"""
        if not self.is32bits:
            if self.is_signed:
                self.val1 = as_int16_t(value[self.start_p:], self.length)
            else:
                self.val2 = as_uint16_t(value[self.start_p:], self.length)
        else:
            if self.is_signed:
                self.val1 = as_int32_t(value[self.start_p:], self.length)
            else:
                self.val2 = as_uint32_t(value[self.start_p:], self.length)
        return 0
    
    def get_value(self) -> bytes:
        """Get value as bytes"""
        if self.is_signed:
            if self.is32bits:
                return to_4_bytes_from_int32_t(self.val1)
            else:
                return to_2_bytes_from_int16_t(self.val1)
        else:
            if self.is32bits:
                return to_4_bytes_from_uint32_t(self.val2)
            else:
                return to_2_bytes_from_uint16_t(self.val2)
    
    def get_float_value(self) -> float:
        """Get value as float"""
        return float(self.val1 if self.is_signed else self.val2)


class NothingTodoModel(AcuratorModel):
    """Model that does nothing"""
    
    def set_value(self, value: bytes) -> int:
        """Do nothing"""
        return 0
    
    def get_value(self) -> bytes:
        """Return empty bytes"""
        return b''
    
    def solve_value(self, output: List[float]):
        """Do nothing"""
        return


# Actuator Models
class NullActuator(NothingTodoModel):
    """Null actuator model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure null actuator"""
        self.start_p = start_p
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class MoveDevice(CommonParsedValueModel):
    """Move device model with forward/backward polynomial mapping"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
        self.forward_poly: List[float] = []
        self.backward_poly: List[float] = []
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure move device"""
        self.start_p = start_p
        self.length = 2
        self.is32bits = False
        self.is_signed = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        
        for p in param:
            if p.name == "ForwardPoly":
                parts = p.value.split(",")
                if len(parts) >= 2:
                    self.forward_poly.append(float(parts[0]))  # high
                    self.forward_poly.append(float(parts[1]))  # low
            elif p.name == "BackwardPoly":
                parts = p.value.split(",")
                if len(parts) >= 2:
                    self.backward_poly.append(float(parts[0]))  # high
                    self.backward_poly.append(float(parts[1]))  # low
            else:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, output: List[float]):
        """Solve move device value"""
        input_val = self.get_float_value()
        t_input = input_val
        
        max_ctrl = self.extra_val.get("MaxControl", 0.0)
        min_ctrl = self.extra_val.get("MinControl", 0.0)
        
        if input_val > 0:
            high = self.forward_poly[0] if len(self.forward_poly) >= 2 else 1.0
            low = self.forward_poly[1] if len(self.forward_poly) >= 2 else 0.0
        else:
            high = self.backward_poly[0] if len(self.backward_poly) >= 2 else 1.0
            low = self.backward_poly[1] if len(self.backward_poly) >= 2 else 0.0
        
        if t_input > max_ctrl:
            t_input = max_ctrl
        elif t_input < min_ctrl:
            t_input = min_ctrl
        
        if t_input == 0.0:
            output.append(0.0)
            return
        
        result = (t_input - low) / high
        output.append(result)


class SteeringDevice(CommonParsedValueModel):
    """Steering device model with polynomial mapping"""
    
    def __init__(self):
        super().__init__()
        self.poly: List[float] = []
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure steering device"""
        self.start_p = start_p
        self.length = 2
        self.is_signed = True
        self.is32bits = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        
        for p in param:
            if p.name == "Poly":
                parts = p.value.split(",")
                if len(parts) >= 2:
                    self.poly.append(float(parts[0]))  # high
                    self.poly.append(float(parts[1]))  # low
    
    def solve_value(self, output: List[float]):
        """Solve steering device value"""
        input_val = self.get_float_value()
        low = self.poly[1] if len(self.poly) >= 2 else 0.0
        high = self.poly[0] if len(self.poly) >= 2 else 1.0
        result = (input_val - low) / high
        output.append(result)


class ForkDevice(CommonParsedValueModel):
    """Fork device model with positive/negative polynomial mapping"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
        self.positive_poly: List[float] = []
        self.negative_poly: List[float] = []
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure fork device"""
        self.start_p = start_p
        self.length = 2
        self.is32bits = False
        self.is_signed = True
        if length != -1:
            self.length = length
        self.is_need_parsed = True
        self.end_p = self.start_p + self.length - 1
        
        for p in param:
            if p.name == "PositivePoly":
                parts = p.value.split(",")
                if len(parts) >= 2:
                    self.positive_poly.append(float(parts[0]))  # high
                    self.positive_poly.append(float(parts[1]))  # low
            elif p.name == "NegativePoly":
                parts = p.value.split(",")
                if len(parts) >= 2:
                    self.negative_poly.append(float(parts[0]))  # high
                    self.negative_poly.append(float(parts[1]))  # low
            else:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, output: List[float]):
        """Solve fork device value"""
        input_val = self.get_float_value()
        t_input = input_val
        
        max_ctrl = self.extra_val.get("MaxControl", 0.0)
        min_ctrl = self.extra_val.get("MinControl", 0.0)
        
        if input_val >= 0:
            high = self.positive_poly[0] if len(self.positive_poly) >= 2 else 1.0
            low = self.positive_poly[1] if len(self.positive_poly) >= 2 else 0.0
        else:
            high = self.negative_poly[0] if len(self.negative_poly) >= 2 else 1.0
            low = self.negative_poly[1] if len(self.negative_poly) >= 2 else 0.0
        
        if t_input >= max_ctrl:
            t_input = max_ctrl
        elif t_input <= min_ctrl:
            t_input = min_ctrl
        
        if t_input == 0.0:
            output.append(0.0)
            return
        
        result = (t_input - low) / high
        output.append(result)


class LiftDevice(NothingTodoModel):
    """Lift device model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure lift device"""
        self.start_p = start_p
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class SwitchActuator(CommonParsedValueModel):
    """Switch actuator model for bit manipulation"""
    
    def __init__(self):
        super().__init__()
        self.bitmap: List[bool] = []
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure switch actuator"""
        self.start_p = start_p
        self.length = 10
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.bitmap = [False] * (self.length * 8)
        
        for p in param:
            try:
                bit_pos = int(p.value)
                if 0 <= bit_pos < len(self.bitmap):
                    self.bitmap[bit_pos] = False
            except ValueError:
                pass
        
        self.buf = bytearray(self.length)
    
    def set_value(self, value: bytes) -> int:
        """Set value from bytes"""
        self.buf[:self.length] = value[self.start_p:self.start_p + self.length]
        return 0
    
    def get_value(self) -> bytes:
        """Get value as bytes"""
        return bytes(self.buf)
    
    def solve_value(self, output: List[float]):
        """Do nothing for switch actuator"""
        return


class ADataIndex(CommonParsedValueModel):
    """Data index model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure data index"""
        self.start_p = start_p
        self.length = 4
        self.is32bits = True
        self.is_signed = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
    
    def solve_value(self, output: List[float]):
        """Do nothing for data index"""
        return


class ADataIgnore(NothingTodoModel):
    """Data ignore model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure data ignore"""
        self.start_p = start_p
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class MCUDataIndexReturn(NothingTodoModel):
    """MCU data index return model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure MCU data index return"""
        self.start_p = start_p
        self.length = 4
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class ValveControleDevice(NothingTodoModel):
    """Valve control device model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure valve control device"""
        self.start_p = start_p
        self.length = 0
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class SerialDataActuator(NothingTodoModel):
    """Serial data actuator model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure serial data actuator"""
        self.start_p = start_p
        self.length = 0
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class GPIOSwitchActuator(NothingTodoModel):
    """GPIO switch actuator model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure GPIO switch actuator"""
        self.start_p = start_p
        self.length = 0
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


class CanToWifiActuator(NothingTodoModel):
    """CAN to WiFi actuator model"""
    
    def config(self, param: List[SpecialParam], start_p: int, length: int = -1, is32bits: bool = False):
        """Configure CAN to WiFi actuator"""
        self.start_p = start_p
        self.length = 0
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1


# Sensor Models
class SensorModel(ABC):
    """Base class for sensor models"""
    
    def __init__(self):
        self.start_p: int = 0
        self.length: int = 0
        self.end_p: int = 0
        self.is_need_parsed: bool = False
        self.type: str = "undefined"
        self.buf: Optional[bytearray] = None
    
    def get_length(self) -> int:
        """Get length of the model"""
        return self.length
    
    def get_buf(self) -> bytes:
        """Get buffer as bytes"""
        return bytes(self.buf) if self.buf else b''
    
    def get_start_p(self) -> int:
        """Get start position"""
        return self.start_p
    
    def get_end_p(self) -> int:
        """Get end position"""
        return self.end_p
    
    def is_need_parsed(self) -> bool:
        """Check if model needs parsing"""
        return self.is_need_parsed
    
    def sensor_model_config(self):
        """Configure sensor model buffer"""
        if self.buf is None:
            self.buf = bytearray(self.length)
            for i in range(self.length):
                self.buf[i] = 0x00
    
    def get_type(self) -> str:
        """Get data type string"""
        return self.type
    
    def _set_type(self, is_signed: bool = False):
        """Set data type based on length and signedness"""
        if self.length == 2:
            self.type = "int16_t" if is_signed else "uint16_t"
        elif self.length == 4:
            self.type = "int32_t" if is_signed else "uint32_t"
        else:
            self.type = "undefined"
    
    def print_info(self):
        """Print model information (for debugging)"""
        pass
    
    @abstractmethod
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure the model"""
        pass
    
    @abstractmethod
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve value based on input"""
        pass
    
    @abstractmethod
    def set_value(self, val: List['UpdateValue']) -> int:
        """Set value from update values"""
        pass


class CommonParsedSencerModel(SensorModel):
    """Common parsed sensor model"""
    
    def __init__(self):
        super().__init__()
        self.is_signed: bool = False
        self.is32bits: bool = False
    
    def set_value(self, val: List['UpdateValue']) -> int:
        """Set value from update values"""
        if not self.is32bits:
            if self.is_signed:
                l_val = val[0].val
                to_2_bytes_from_int16_t_buf(l_val, self.buf)
            else:
                l_val = val[0].val
                to_2_bytes_from_uint16_t_buf(l_val, self.buf)
        else:
            if self.is_signed:
                l_val = val[0].val
                to_4_bytes_from_int32_t_buf(l_val, self.buf)
            else:
                # print("val : ", val)
                l_val = val[0].val
                to_4_bytes_from_uint32_t_buf(l_val, self.buf)
        return 0


class CommonNothingTodoSencerModel(SensorModel):
    """Common nothing to do sensor model"""
    
    def set_value(self, val: List['UpdateValue']) -> int:
        """Do nothing"""
        return 0
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing"""
        return


# Sensor Model Implementations
class DataHeader(CommonNothingTodoSencerModel):
    """Data header model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure data header"""
        self.start_p = last
        self.length = 1
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self.buf[0] = 0xAA


class SDataIgnore(CommonNothingTodoSencerModel):
    """Sensor data ignore model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure sensor data ignore"""
        self.start_p = last
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()


class WheelCoder(CommonParsedSencerModel):
    """Wheel coder model for left/right wheel encoding"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure wheel coder"""
        self.start_p = last
        self.length = 8  # left, right
        self.is_signed = False
        self.is32bits = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(False)
        
        for p in param:
            self.extra_val[p.name] = float(p.value)

    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve wheel coder value"""
        # 0 is left, 1 is right
        lscale = self.extra_val.get("LeftScale", 1.0)
        rscale = self.extra_val.get("RightScale", 1.0)
        magnification = self.extra_val.get("Magnification", 1.0)
        
        output.append(input_vals[0] / magnification / lscale)
        output.append(input_vals[1] / magnification / rscale)
        
        # Convert to bytes
        l_wheel = int(output[0])
        r_wheel = int(output[1])
        to_4_bytes_from_uint32_t_buf(l_wheel, self.buf)
        to_4_bytes_from_uint32_t_buf(r_wheel, self.buf, self.length // 2)
    
    def set_value(self, val: List['UpdateValue']) -> int:
        """Set value for wheel coder"""
        if not self.is32bits:
            if self.is_signed:
                l_val = val[0].val  # left
                to_2_bytes_from_int16_t_buf(l_val, self.buf)
                l_val = val[1].val  # right
                to_2_bytes_from_int16_t_buf(l_val, self.buf, self.length // 2)
            else:
                l_val = val[0].val  # left
                to_2_bytes_from_uint16_t_buf(l_val, self.buf)
                l_val = val[1].val  # right
                to_2_bytes_from_uint16_t_buf(l_val, self.buf, self.length // 2)
        else:
            if self.is_signed:
                l_val = val[0].val  # left
                to_4_bytes_from_int32_t_buf(l_val, self.buf)
                l_val = val[1].val  # right
                to_4_bytes_from_int32_t_buf(l_val, self.buf, self.length // 2)
            else:
                l_val = val[0].val  # right
                to_4_bytes_from_uint32_t_buf(l_val, self.buf)
                l_val = val[1].val  # right
                to_4_bytes_from_uint32_t_buf(l_val, self.buf, self.length // 2)
        return 0


class BatterySencer(CommonParsedSencerModel):
    """Battery sensor model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure battery sensor"""
        self.start_p = last
        self.length = 2
        self.is32bits = False
        self.is_signed = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self._set_type(False)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing for battery sensor"""
        return


class IncrementalSteeringCoder(CommonParsedSencerModel):
    """Incremental steering coder model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
        self.str_val: Dict[str, str] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure incremental steering coder"""
        self.start_p = last
        self.length = 2
        self.is32bits = False
        self.is_signed = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(True)
        
        for p in param:
            if p.name == "RareUp" or p.name == "DisplayName" or p.name == 'Function':
                self.str_val[p.name] = p.value
            else:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve incremental steering coder value"""
        steering_zero = self.extra_val.get("SteeringZero", 0.0)
        magnification = self.extra_val.get("Magnification", 1.0)
        
        result = steering_zero + input_vals[0] / magnification
        output.append(result)
        
        incremental_coder = int(result)
        to_2_bytes_from_int16_t_buf(incremental_coder, self.buf)


class Gyroscope(CommonParsedSencerModel):
    """Gyroscope model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure gyroscope"""
        self.start_p = last
        self.length = 4
        self.is_signed = True
        self.is32bits = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(False)
        
        for p in param:
            if p.name == "Function" or p.name == "Signed" or p.name == "DisplayName":
                continue
                # self.extra_val[p.name] = p.value
            else:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve gyroscope value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        result = input_vals[0] / magnification
        output.append(result)
        
        gyro = int(result)
        to_4_bytes_from_int32_t_buf(gyro, self.buf)


class ElePerceptionCameraDistance(CommonParsedSencerModel):
    """Electronic perception camera distance model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure electronic perception camera distance"""
        self.start_p = last
        self.length = 2
        
        for p in param:
            if p.name == "Signed":
                self.is_signed = p.value.lower() == "true"
            elif p.name == "Is32Bit":
                self.is32bits = p.value.lower() == "true"
                if self.is32bits:
                    self.length = 4
        
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve electronic perception camera distance value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        if not self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_uint16_t_buf(height, self.buf)
        elif not self.is_signed and self.is32bits:
            height = int(result)
            to_4_bytes_from_uint32_t_buf(height, self.buf)
        elif self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_int16_t_buf(height, self.buf)
        else:
            height = int(result)
            to_4_bytes_from_int32_t_buf(height, self.buf)


class SDataIndex(CommonParsedSencerModel):
    """Sensor data index model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure sensor data index"""
        self.start_p = last
        self.length = 4
        self.is_signed = False
        self.is32bits = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self._set_type(False)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing for sensor data index"""
        return


class ForkDisplacementSencer(CommonParsedSencerModel):
    """Fork displacement sensor model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure fork displacement sensor"""
        self.start_p = last
        self.length = 2
        
        for p in param:
            if p.name == "Signed":
                self.is_signed = p.value.lower() == "true"
            elif p.name == "Is32Bit":
                self.is32bits = p.value.lower() == "true"
                if self.is32bits:
                    self.length = 4
        
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            if p.name == "Function" or p.name == "Signed" or p.name == "Is32Bit":
                continue
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve fork displacement sensor value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        if not self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_uint16_t_buf(height, self.buf)
        elif not self.is_signed and self.is32bits:
            height = int(result)
            to_4_bytes_from_uint32_t_buf(height, self.buf)
        elif self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_int16_t_buf(height, self.buf)
        else:
            height = int(result)
            to_4_bytes_from_int32_t_buf(height, self.buf)


class HeightCoder(CommonParsedSencerModel):
    """Height coder model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure height coder"""
        self.start_p = last
        self.length = 4
        self.is_signed = True
        self.is32bits = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(False)
        
        for p in param:
            if p.name == "Is32Bit" or p.name == "Signed":
                continue
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve height coder value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        height_coder = int(result)
        to_4_bytes_from_int32_t_buf(height_coder, self.buf)


class HolzerCoder(CommonParsedSencerModel):
    """Holzer coder model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure Holzer coder"""
        self.start_p = last
        self.length = 2
        self.is_signed = False
        self.is32bits = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(False)
        
        for p in param:
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve Holzer coder value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        result = input_vals[0] / magnification
        output.append(result)
        
        holzer = int(result)
        to_2_bytes_from_uint16_t_buf(holzer, self.buf)


class DataIndexReturn(CommonParsedSencerModel):
    """Data index return model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure data index return"""
        self.start_p = last
        self.length = 4
        self.is32bits = True
        self.is_signed = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self._set_type(False)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing for data index return"""
        return


class NullSencer(CommonNothingTodoSencerModel):
    """Null sensor model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure null sensor"""
        self.start_p = last
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()


class ErrorCode(CommonNothingTodoSencerModel):
    """Error code model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure error code"""
        self.start_p = last
        self.length = 4
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()


class RPMSensor(CommonParsedSencerModel):
    """RPM sensor model"""
    
    def __init__(self):
        super().__init__()
        self.magnification: float = 1.0
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure RPM sensor"""
        self.start_p = last
        self.length = 2
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(is_signed)
        
        for p in param:
            if p.name == "Magnification":
                self.magnification = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve RPM sensor value"""
        result = input_vals[0] / self.magnification
        output.append(result)
        
        motor = int(result)
        to_2_bytes_from_uint16_t_buf(motor, self.buf)


class VelocityControlLevel(CommonNothingTodoSencerModel):
    """Velocity control level model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure velocity control level"""
        self.start_p = last
        self.length = 1
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self.buf[0] = 0x07


class SwitchSencer(SensorModel):
    """Switch sensor model for bit manipulation"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure switch sensor"""
        self.start_p = last
        self.length = 12
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing for switch sensor"""
        return
    
    def set_value(self, val: List['UpdateValue']) -> int:
        """Set switch sensor value"""
        v = val[0].val
        bytes_index = val[0].subId // 8
        shift = val[0].subId - bytes_index * 8
        
        if v:
            self.buf[bytes_index] |= (0x01 << shift)
        else:
            self.buf[bytes_index] &= ~(0x01 << shift)
        return 0


class Accelerometer(CommonParsedSencerModel):
    """Accelerometer model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure accelerometer"""
        self.start_p = last
        self.length = 2
        self.is_signed = True
        self.is_need_parsed = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            if p.name in ["Zero", "Magnification"]:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve accelerometer value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        accelerometer = int(result)
        to_2_bytes_from_int16_t_buf(accelerometer, self.buf)


class AngularVelocitySensor(CommonParsedSencerModel):
    """Angular velocity sensor model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure angular velocity sensor"""
        self.start_p = last
        self.length = 2
        self.is_signed = True
        self.is_need_parsed = True
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        
        for p in param:
            if p.name in ["Zero", "Magnification"]:
                self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve angular velocity sensor value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        angular_velocity = int(result)
        to_2_bytes_from_int16_t_buf(angular_velocity, self.buf)


class HydraulicPressureSensor(CommonParsedSencerModel):
    """Hydraulic pressure sensor model"""
    
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure hydraulic pressure sensor"""
        self.start_p = last
        self.length = 2
        
        for p in param:
            if p.name == "Signed":
                self.is_signed = p.value.lower() == "true"
            elif p.name == "Is32Bit":
                self.is32bits = p.value.lower() == "true"
                if self.is32bits:
                    self.length = 4
        
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve hydraulic pressure sensor value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        if not self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_uint16_t_buf(height, self.buf)
        elif not self.is_signed and self.is32bits:
            height = int(result)
            to_4_bytes_from_uint32_t_buf(height, self.buf)
        elif self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_int16_t_buf(height, self.buf)
        else:
            height = int(result)
            to_4_bytes_from_int32_t_buf(height, self.buf)


class DataCRC(CommonParsedSencerModel):
    """Data CRC model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure data CRC"""
        self.start_p = last
        self.length = 2
        self.is32bits = False
        self.is_signed = False
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Do nothing for data CRC"""
        return


class DataTail(CommonNothingTodoSencerModel):
    """Data tail model"""
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure data tail"""
        self.start_p = last
        self.length = 1
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.sensor_model_config()
        self.buf[0] = 0x55


class LaserDistance(CommonNothingTodoSencerModel):
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure fork displacement sensor"""
        self.start_p = last
        self.length = 2
        
        for p in param:
            if p.name == "Signed":
                self.is_signed = p.value.lower() == "true"
            elif p.name == "Is32Bit":
                self.is32bits = p.value.lower() == "true"
                if self.is32bits:
                    self.length = 4
        
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            if p.name == "Function" or p.name == "Signed" or p.name == "Is32Bit" or p.name == "DisplayName":
                continue
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve fork displacement sensor value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        if not self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_uint16_t_buf(height, self.buf)
        elif not self.is_signed and self.is32bits:
            height = int(result)
            to_4_bytes_from_uint32_t_buf(height, self.buf)
        elif self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_int16_t_buf(height, self.buf)
        else:
            height = int(result)
            to_4_bytes_from_int32_t_buf(height, self.buf)

class WeightSensor(CommonNothingTodoSencerModel):
    def __init__(self):
        super().__init__()
        self.extra_val: Dict[str, float] = {}
    
    def config(self, param: List[SpecialParam], last: int, length: int = -1, is32bits: bool = False, is_signed: bool = False):
        """Configure fork displacement sensor"""
        self.start_p = last
        self.length = 2
        
        for p in param:
            if p.name == "Signed":
                self.is_signed = p.value.lower() == "true"
            elif p.name == "Is32Bit":
                self.is32bits = p.value.lower() == "true"
                if self.is32bits:
                    self.length = 4
        
        if length != -1:
            self.length = length
        self.end_p = self.start_p + self.length - 1
        self.is_need_parsed = True
        self.sensor_model_config()
        self._set_type(self.is_signed)
        
        for p in param:
            if p.name == "Function" or p.name == "Signed" or p.name == "Is32Bit" or p.name == "DisplayName":
                continue
            self.extra_val[p.name] = float(p.value)
    
    def solve_value(self, input_vals: List[float], output: List[float]):
        """Solve fork displacement sensor value"""
        magnification = self.extra_val.get("Magnification", 1.0)
        zero = self.extra_val.get("Zero", 0.0)
        result = input_vals[0] / magnification + zero
        output.append(result)
        
        if not self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_uint16_t_buf(height, self.buf)
        elif not self.is_signed and self.is32bits:
            height = int(result)
            to_4_bytes_from_uint32_t_buf(height, self.buf)
        elif self.is_signed and not self.is32bits:
            height = int(result)
            to_2_bytes_from_int16_t_buf(height, self.buf)
        else:
            height = int(result)
            to_4_bytes_from_int32_t_buf(height, self.buf)

# Import UpdateValue here to avoid circular import
# from parser import UpdateValue
