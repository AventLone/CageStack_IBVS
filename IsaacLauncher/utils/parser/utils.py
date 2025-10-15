"""
Utility functions for byte manipulation and CRC calculation
Converted from C++ utils.hpp and utils.cpp
"""

import struct
from typing import Union


def as_int16_t(buf: bytes, length: int) -> int:
    """Convert bytes to int16_t (big-endian)"""
    if len(buf) < 2:
        return 0
    return struct.unpack('>h', buf[:2])[0]


def as_uint16_t(buf: bytes, length: int) -> int:
    """Convert bytes to uint16_t (big-endian)"""
    if len(buf) < 2:
        return 0
    return struct.unpack('>H', buf[:2])[0]


def as_int32_t(buf: bytes, length: int) -> int:
    """Convert bytes to int32_t (big-endian)"""
    if len(buf) < 4:
        return 0
    return struct.unpack('>i', buf[:4])[0]


def as_uint32_t(buf: bytes, length: int) -> int:
    """Convert bytes to uint32_t (big-endian)"""
    if len(buf) < 4:
        return 0
    return struct.unpack('>I', buf[:4])[0]


def to_2_bytes_from_int16_t(val: int) -> bytes:
    """Convert int16_t to 2 bytes (big-endian)"""
    return struct.pack('>h', val)


def to_2_bytes_from_uint16_t(val: int) -> bytes:
    """Convert uint16_t to 2 bytes (big-endian)"""
    return struct.pack('>H', val)


def to_4_bytes_from_int32_t(val: int) -> bytes:
    """Convert int32_t to 4 bytes (big-endian)"""
    return struct.pack('>i', val)


def to_4_bytes_from_uint32_t(val: int) -> bytes:
    """Convert uint32_t to 4 bytes (big-endian)"""
    return struct.pack('>I', val)


def cal_crc(buffer: bytes, offset: int, length: int, w_crc: int, w_polynom: int) -> int:
    """
    Calculate CRC checksum
    
    Args:
        buffer: Input buffer
        offset: Start offset
        length: Length to process
        w_crc: Initial CRC value
        w_polynom: Polynomial value
        
    Returns:
        Calculated CRC value
    """
    crc = w_crc
    
    for i in range(length):
        if offset + i < len(buffer):
            crc ^= buffer[offset + i]
            for j in range(8):
                if (crc & 0x0001) == 0x0001:
                    crc = (crc >> 1) ^ w_polynom
                else:
                    crc = crc >> 1
    
    return crc & 0xFFFF  # Ensure 16-bit result


# Convenience functions for direct buffer manipulation
def to_2_bytes_from_int16_t_buf(val: int, buf: bytearray, offset: int = 0):
    """Convert int16_t to 2 bytes and store in buffer"""
    data = to_2_bytes_from_int16_t(val)
    buf[offset:offset+2] = data


def to_2_bytes_from_uint16_t_buf(val: int, buf: bytearray, offset: int = 0):
    """Convert uint16_t to 2 bytes and store in buffer"""
    data = to_2_bytes_from_uint16_t(val)
    buf[offset:offset+2] = data


def to_4_bytes_from_int32_t_buf(val: int, buf: bytearray, offset: int = 0):
    """Convert int32_t to 4 bytes and store in buffer"""
    data = to_4_bytes_from_int32_t(val)
    buf[offset:offset+4] = data


def to_4_bytes_from_uint32_t_buf(val: int, buf: bytearray, offset: int = 0):
    """Convert uint32_t to 4 bytes and store in buffer"""
    data = to_4_bytes_from_uint32_t(val)
    buf[offset:offset+4] = data
