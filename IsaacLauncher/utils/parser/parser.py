"""
Main parser classes for InputDecoder and OutputEncoder
Converted from C++ parser.hpp and parser.cpp
"""

import xml.etree.ElementTree as ET
from typing import List, Optional, Union, Any
from dataclasses import dataclass
from .directors import ActuatorDirector, SencerDirector, Package, UpdateValue
from ..logger_utils import LoggerUtil

# 创建日志实例
logger = LoggerUtil.get_logger("parser")

def _convert_bool_string(value: Union[str, int]) -> Union[str, int]:
    """
    Convert boolean string to numeric value if needed.
    
    Args:
        value: String or integer value to convert
        
    Returns:
        Converted value - either original string or numeric 0/1
    """
    # If value is already an integer, return it as-is
    if isinstance(value, int):
        return value
    
    # If value is a string, check if it's a boolean string
    try:
        if value.lower() == 'true':
            return 1
        elif value.lower() == 'false':
            return 0
    except Exception as e:
        # 记录异常日志，包含堆栈信息
        LoggerUtil.log_exception(logger, f"Failed to convert value '{value}' to lowercase", e)
        return value
    return value


@dataclass
class SpecialParam:
    """Special parameter structure"""
    name: str
    value: str


class InputDecoder:
    """Input decoder for parsing actuator data from XML config"""
    
    def __init__(self):
        self._ad_ptr: Optional[ActuatorDirector] = None
    
    def load_config(self, path: str) -> int:
        """
        Load Actuators.config
        
        Args:
            path: Path to the config file
            
        Returns:
            Error code (0 = successful, -1 = config not loaded)
        """
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            if self._ad_ptr is None:
                self._ad_ptr = ActuatorDirector()
            
            is_over = False
            is_start = False
            param_list = []
            
            for ele in root:
                if is_over:
                    break

                name = ele.tag
                if name == "DataHeader":
                    is_start = True

                if not is_start:
                    continue

                function = ""
                length = -1
                param = SpecialParam("", "")

                for attr_name, attr_value in ele.attrib.items():
                    if attr_name == "Length":
                        length = int(attr_value)
                    else:
                        param.name = attr_name
                        param.value = attr_value
                        if param.name == "Function":
                            function = param.value
                        else:
                            param_list.append(SpecialParam(param.name, param.value))

                print("InputEncoder : ", name, function, param_list, length)    
                self._ad_ptr.create(name, function, param_list, length)
                param_list.clear()

                if name == "DataTail":
                    is_over = True
                    break
            
            return True
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def decode_package(self, package: Package) -> int:
        """
        Parse frame
        
        Args:
            package: Frame info
            
        Returns:
            Error code (0 = successful, -1 = config not loaded)
        """
        if self._ad_ptr is None:
            return -1
        
        self._ad_ptr.decode(package.buf, package.len)
        return 0
    
    def get_value(self, key: str, output: List[float], func_key: str = "") -> int:
        """
        Transform value for each node based on Actuators.config
        
        Args:
            key: Key name
            output: Output value list
            func_key: Function key
            
        Returns:
            Error code (0 = successful, -1 = config not loaded, -2 = not parsed)
        """
        if self._ad_ptr is None:
            return -1
        
        handle = self._ad_ptr.get(key, func_key)
        # print("_ad_ptr : ", self._ad_ptr)
        # print("handle : ", handle)
        ptr = self._ad_ptr.get_model(handle)
        # print("ptr : ", ptr)
        
        if ptr is not None and ptr.is_need_parsed:
            ptr.solve_value(output)
            return 0
        else:
            return -2
    
    def get_value2(self, key: str, output: bytearray, size: int, func_key: str = "") -> int:
        """
        Get value from origin bytes (uint16_t, int16_t, uint32_t, int32_t)
        
        Args:
            key: Key name
            output: Output buffer
            size: Size of the valuable
            func_key: Function key
            
        Returns:
            Error code
        """
        if self._ad_ptr is None:
            return -1
        
        handle = self._ad_ptr.get(key, func_key)
        ptr = self._ad_ptr.get_model(handle)
        length = ptr.get_length()
        # print("length : ", length, " ptr.get_value() : ", ptr.get_value())
        output[:length] = ptr.get_value()[:length]
        return 0
    
    def get_switch_value(self, key: str, bits: int, output: List[bool], func_key: str = "") -> int:
        """
        Get value from switch bits value
        
        Args:
            key: Key name
            bits: Bit position
            output: Output boolean list
            func_key: Function key
            
        Returns:
            Error code
        """
        if self._ad_ptr is None:
            return -1
        
        handle = self._ad_ptr.get(key, func_key)
        ptr = self._ad_ptr.get_model(handle)
        header = ptr.get_value()
        length = ptr.get_length()
        
        bytes_index = bits // 8
        shift = bits - bytes_index * 8
        
        if bytes_index < length:
            output[0] = bool(header[bytes_index] & (0x01 << shift))
        else:
            output[0] = False
        
        return 0


class OutputEncoder:
    """Output encoder for creating sensor data packages"""
    
    def __init__(self):
        self._sd_ptr: Optional[SencerDirector] = None
    
    def load_config(self, path: str) -> int:
        """
        Load Sencers.config
        
        Args:
            path: Path to the config file
            
        Returns:
            Error code (0 = successful, -1 = config not loaded, -2 = already loaded)
        """
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            if self._sd_ptr is not None:
                return -2
            
            self._sd_ptr = SencerDirector()
            
            is_over = False
            is_start = False
            param_list = []
            
            for ele in root:
                if is_over:
                    break

                name = ele.tag
                function = ""
                length = -1
                param = SpecialParam("", "")
                    
                if name == "DataHeader":
                    is_start = True

                if not is_start:
                    continue

                for attr_name, attr_value in ele.attrib.items():
                    if attr_name == "Length":
                        length = int(attr_value)
                    else:
                        param.name = attr_name
                        param.value = attr_value
                        param_list.append(SpecialParam(param.name, param.value))
                        if param.name == "Function":
                            function = param.value

                print("OutputEncoder : ", name, function, param_list, length)
                self._sd_ptr.create(name, function, param_list, length)
                param_list.clear()

                if name == "DataTail":
                    is_over = True
                    break
            
            return True
            
        except Exception as e:
            LoggerUtil.log_exception(logger, f"Failed to load config file '{path}'", e)
            return True
    
    def update_value(self, key: str, len: int, func_key: str = "", *args) -> int:
        """
        Transform value for each node based on Sensors.config
        
        Args:
            key: Key name
            len: Number of input values
            func_key: Function key
            *args: Variable number of double values
            
        Returns:
            Error code (0 = successful, -1 = config not loaded, -2 = not parsed)
        """
        if self._sd_ptr is None:
            return -1
        
        calculated_input = list(args[:len])
        calculated_output = []
        
        handle = self._sd_ptr.get(key, func_key)
        # print("handle : ", handle)
        ptr = self._sd_ptr.get_model(handle)
        # print("get_model: ", ptr)
        
        if ptr is not None and ptr.is_need_parsed:
            ptr.solve_value(calculated_input, calculated_output)
            return 0
        else:
            return -2
    
    def update_switch_value(self, key: str, bits: int, value: bool, func_key: str = "") -> int:
        """
        Update bits value
        
        Args:
            key: Key name
            bits: Bit position
            value: Boolean value
            func_key: Function key
            
        Returns:
            Error code (0 = successful, -1 = invalid key, -2 = invalid bits position)
        """
        if self._sd_ptr is None:
            return -1
        
        ll_input = UpdateValue(val=value, len=1, subId=bits)
        l_input = [ll_input]
        
        handle = self._sd_ptr.get(key, func_key)
        ptr = self._sd_ptr.get_model(handle)
        ptr.set_value(l_input)
        
        return 0
    
    def update_value2(self, key: str, input_data: bytes, size: int, func_key: str = "") -> int:
        """
        Update bytes value to each node
        
        Args:
            key: Key name
            input_data: Input data bytes
            size: Size of input data
            func_key: Function key
            
        Returns:
            Error code
        """
        if self._sd_ptr is None:
            return -1
        
        ll_input = UpdateValue(val=input_data, len=size, subId=0)
        l_input = [ll_input]
        
        handle = self._sd_ptr.get(key, func_key)
        # print(handle)
        ptr = self._sd_ptr.get_model(handle)
        # print(ptr)
        ptr.set_value(l_input)
        
        return 0
    
    def encode_package(self) -> Optional[Package]:
        """
        Get n bytes frame
        
        Returns:
            Frame info or None if failed
        """
        if self._sd_ptr is None:
            return None
        
        return self._sd_ptr.get_package()


