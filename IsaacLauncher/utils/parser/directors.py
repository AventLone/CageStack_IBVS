"""
Director classes for managing actuator and sensor models
Converted from C++ ModelDirector.hpp and ModelDirector.cpp
"""

from typing import List, Dict, Optional, Any
from .models import *
from .utils import cal_crc

@dataclass
class Package:
    """Data package structure"""
    buf: bytes
    len: int


@dataclass
class UpdateValue:
    """Update value structure for sensor data"""
    val: Any
    len: int  # size of the bytes
    subId: int  # subId for WheelCoder (0=left, 1=right), or any value if <=1 params


class ActuatorDirector:
    """Director class for managing actuator models"""
    
    def __init__(self):
        self.total: int = 0
        self.package_set: List[AcuratorModel] = []
        self.support_handle: Dict[str, int] = {}
    
    def create(self, name: str, function: str, param: List[SpecialParam], length: int = -1, is32bits: bool = False):
        """Create an actuator model based on name"""
        ptr: Optional[AcuratorModel] = None
        is_support_modified = True
        
        if name == "MoveDevice":
            ptr = MoveDevice()
        elif name == "SteeringDevice":
            ptr = SteeringDevice()
        elif name == "ForkDevice":
            ptr = ForkDevice()
        elif name == "LiftDevice":
            ptr = LiftDevice()
        elif name == "SwitchActuator":
            ptr = SwitchActuator()
        elif name == "DataIndex":
            ptr = ADataIndex()
        elif name == "NullActuator":
            ptr = NullActuator()
            is_support_modified = False
        elif name == "MCUDataIndexReturn":
            ptr = MCUDataIndexReturn()
        else:
            ptr = ADataIgnore()
            is_support_modified = False
        
        # Calculate end position
        end_p = 0
        if self.package_set:
            end_p = self.package_set[-1].get_end_p() + 1
        
        # Configure the model
        ptr.config(param, end_p, length, is32bits)
        self.package_set.append(ptr)
        self.total += 1
        
        # Add to support handle if modifiable
        combined_key = name + function
        if is_support_modified:
            self.support_handle[combined_key] = self.total - 1
    
    def decode(self, buf: bytes, length: int):
        """Decode buffer data for all actuator models"""
        for ele in self.package_set:
            ele.set_value(buf)
    
    def get(self, name: str, func_key: str) -> int:
        """Get handle for a model by name and function key"""
        combined_key = name + func_key
        # print("combined_key : ", combined_key)
        # print("self.support_handle : ", self.support_handle)
        return self.support_handle.get(combined_key, -1)
    
    def get_model(self, handle: int) -> Optional[AcuratorModel]:
        """Get model by handle"""
        # print("self.package_set : ", self.package_set)
        if 0 <= handle < len(self.package_set):
            return self.package_set[handle]
        return None


class SencerDirector:
    """Director class for managing sensor models"""
    
    def __init__(self):
        self.total: int = 0
        self.crc_handle: int = 0
        self.package: Package = Package(buf=b'', len=0)
        self.package_set: List[SensorModel] = []
        self.support_handle: Dict[str, int] = {}
    
    def create(self, name: str, function: str, param: List[SpecialParam], length: int = -1, is32bits: bool = False):
        """Create a sensor model based on name"""
        ptr: Optional[SensorModel] = None
        is_support_modified = True
        
        if name == "WheelCoder":
            ptr = WheelCoder()
        elif name == "BatterySencer":
            ptr = BatterySencer()
        elif name == "IncrementalSteeringCoder":
            ptr = IncrementalSteeringCoder()
        elif name == "Gyroscope":
            ptr = Gyroscope()
        elif name in ["ForkDisplacementSencer", "DisplacementSencer"]:
            ptr = ForkDisplacementSencer()
        elif name == "HeightCoder":
            ptr = HeightCoder()
        elif name == "DataIndex":
            ptr = SDataIndex()
        elif name == "DataIndexReturn":
            ptr = DataIndexReturn()
        elif name == "NullSencer":
            ptr = NullSencer()
            is_support_modified = False
        elif name == "DataHeader":
            ptr = DataHeader()
        elif name == "DataCRC":
            ptr = DataCRC()
        elif name == "DataTail":
            ptr = DataTail()
        elif name == "ErrorCode":
            ptr = ErrorCode()
        elif name == "VelocityControlLevel":
            ptr = VelocityControlLevel()
        elif name == "HolzerCoder":
            ptr = HolzerCoder()
        elif name == "RPMSensor":
            ptr = RPMSensor()
        elif name == "SwitchSencer":
            ptr = SwitchSencer()
        elif name == "Accelerometer":
            ptr = Accelerometer()
        elif name == "AngularVelocitySensor":
            ptr = AngularVelocitySensor()
        elif name == "HydraulicPressureSensor":
            ptr = HydraulicPressureSensor()
        elif name == "ElePerceptionCameraDistance":
            ptr = ElePerceptionCameraDistance()
        elif name == "LaserDistance":
            ptr = LaserDistance()
        elif name == "WeightSensor":
            ptr = WeightSensor()    
        else:
            ptr = SDataIgnore()
            is_support_modified = False
        
        # Calculate end position
        end_p = 0
        if self.package_set:
            end_p = self.package_set[-1].get_end_p() + 1
        
        # Configure the model
        ptr.config(param, end_p, length, is32bits)
        self.package_set.append(ptr)
        self.total += 1
        self.package.len += ptr.get_length()
        
        # Track CRC handle
        if name == "DataCRC":
            self.crc_handle = self.total - 1
        
        # Add to support handle if modifiable
        combined_key = name + function
        if is_support_modified:
            self.support_handle[combined_key] = self.total - 1
    
    def encode(self, handle: int, val_list: List[UpdateValue]):
        """Encode values for a specific sensor model"""
        if 0 <= handle < len(self.package_set):
            self.package_set[handle].set_value(val_list)
    
    def get(self, name: str, function: str) -> int:
        """Get handle for a model by name and function"""
        combined_key = name + function
        # print("self.support_handle : ", self.support_handle)
        return self.support_handle.get(combined_key, -1)
    
    def get_model(self, handle: int) -> Optional[SensorModel]:
        """Get model by handle"""
        if 0 <= handle < len(self.package_set):
            return self.package_set[handle]
        return None
    
    def get_package(self) -> Optional[Package]:
        """Get the complete encoded package"""
        if not self.package_set:
            return None
        
        # Initialize package buffer if needed
        if not self.package.buf:
            self.package.buf = bytearray(self.package.len)
        
        # Copy data from all models
        for it in self.package_set:
            s_p = it.get_start_p()
            e_p = it.get_end_p()
            s_to_e = it.get_length()
            self.package.buf[s_p:s_p + s_to_e] = it.get_buf()[:s_to_e]
        
        # Calculate and set CRC
        if self.crc_handle < len(self.package_set):
            crc_model = self.package_set[self.crc_handle]
            tail_model = self.package_set[-1]
            
            crc_s_p = crc_model.get_start_p()
            crc_s_to_e = crc_model.get_length()
            tail_s_to_e = tail_model.get_length()
            
            # Calculate CRC
            crc = cal_crc(
                self.package.buf, 
                0, 
                self.package.len - tail_s_to_e - crc_s_to_e,
                0xffff, 
                0x8408
            )
            
            # Set CRC value
            crc_val = UpdateValue(val=crc, len=2, subId=0)
            crc_info = [crc_val]
            crc_model.set_value(crc_info)
            
            # Copy CRC to package
            self.package.buf[crc_s_p:crc_s_p + crc_s_to_e] = crc_model.get_buf()[:crc_s_to_e]
        
        return self.package
