"""
Example usage of the Python parser
Converted from C++ main.cpp example
"""

from parser import InputDecoder, OutputEncoder, Package, UpdateValue, SpecialParam


def test_actuator_decoding():
    """Test actuator data decoding"""
    print("=== Testing Actuator Decoding ===")
    
    # Create decoder
    decoder = InputDecoder()
    
    # Load config (you would need actual config files)
    # ret = decoder.load_config("path/to/Actuators.config")
    # if ret != 0:
    #     print(f"Failed to load config: {ret}")
    #     return
    
    # Sample data (from C++ example)
    sample_data = bytes([
        0xAA, 0xFF, 0xFF, 0x00, 0x00, 0x09, 0xA1, 0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xD1, 0x7B, 0x00, 0x0B, 0x7A,
        0xC7, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x7D, 0x55
    ])
    
    package = Package(buf=sample_data, len=len(sample_data))
    
    # Decode package
    ret = decoder.decode_package(package)
    if ret != 0:
        print(f"Failed to decode package: {ret}")
        return
    
    # Get values
    output = []
    ret = decoder.get_value("MoveDevice", output)
    if ret == 0:
        print(f"MoveDevice value: {output[0] if output else 'No value'}")
    else:
        print(f"Failed to get MoveDevice value: {ret}")
    
    # Get switch values
    switch_output = [False]
    for i in range(10 * 8):  # Test first 80 bits
        ret = decoder.get_switch_value("SwitchActuator", i, switch_output)
        if ret == 0 and switch_output[0]:
            print(f"Switch bit {i} is set")


def test_sensor_encoding():
    """Test sensor data encoding"""
    print("\n=== Testing Sensor Encoding ===")
    
    # Create encoder
    encoder = OutputEncoder()
    
    # Load config (you would need actual config files)
    # ret = encoder.load_config("path/to/Sencers.config")
    # if ret != 0:
    #     print(f"Failed to load config: {ret}")
    #     return
    
    # Test data values
    steering_angle = 1.57
    ax = 9.7804
    ay = 0.0344975
    az = 9.7804
    gyro = 3.14
    data_idx = 100
    ll_data_idx = 100
    fork_z = 1.56
    fork_c = 0.1
    fork_y = 0.2
    fork_p = 0.3
    move_wheel = 10.0
    
    # Update values
    encoder.update_value2("DataIndex", data_idx.to_bytes(4, 'big'), 4)
    encoder.update_value2("DataIndexReturn", ll_data_idx.to_bytes(4, 'big'), 4)
    encoder.update_value("WheelCoder", 2, "", move_wheel, move_wheel)
    encoder.update_value("IncrementalSteeringCoder", 1, "LF", steering_angle)
    encoder.update_value("IncrementalSteeringCoder", 1, "RF", steering_angle)
    encoder.update_value("Accelerometer", 1, "X", ax)
    encoder.update_value("Accelerometer", 1, "Y", ay)
    encoder.update_value("Accelerometer", 1, "Z", az)
    encoder.update_value("ForkDisplacementSencer", 1, "Y", fork_y)
    encoder.update_value("ForkDisplacementSencer", 1, "C", fork_c)
    encoder.update_value("ForkDisplacementSencer", 1, "P", fork_p)
    encoder.update_value("HeightCoder", 1, "", fork_z)
    
    # Encode package
    package = encoder.encode_package()
    if package:
        print(f"Encoded package length: {package.len}")
        print("Encoded data:", end=" ")
        for i in range(min(package.len, 20)):  # Show first 20 bytes
            print(f"{package.buf[i]:02X}", end=" ")
        print("..." if package.len > 20 else "")
    else:
        print("Failed to encode package")


def test_switch_operations():
    """Test switch operations"""
    print("\n=== Testing Switch Operations ===")
    
    encoder = OutputEncoder()
    
    # Test switch values
    for i in range(12 * 8):  # 12 bytes = 96 bits
        if i < 32 or i > 33:
            encoder.update_switch_value("SwitchSencer", i, True)
        else:
            encoder.update_switch_value("SwitchSencer", i, False)
    
    print("Switch operations completed")


def create_sample_config():
    """Create sample configuration files for testing"""
    print("\n=== Creating Sample Configuration Files ===")
    
    # Sample actuator config
    actuator_config = """<?xml version="1.0"?>
<Config>
    <Actuators>
        <DataHeader Length="1" Function="Header"/>
        <MoveDevice Length="2" Function="Move" ForwardPoly="1.0,0.0" BackwardPoly="1.0,0.0" MaxControl="100.0" MinControl="-100.0"/>
        <SteeringDevice Length="2" Function="Steer" Poly="1.0,0.0"/>
        <ForkDevice Length="2" Function="Fork" PositivePoly="1.0,0.0" NegativePoly="1.0,0.0" MaxControl="100.0" MinControl="-100.0"/>
        <SwitchActuator Length="10" Function="Switch"/>
        <DataIndex Length="4" Function="Index"/>
        <DataTail Length="1" Function="Tail"/>
    </Actuators>
</Config>"""
    
    # Sample sensor config
    sensor_config = """<?xml version="1.0"?>
<Config>
    <Sensors>
        <DataHeader Length="1" Function="Header"/>
        <WheelCoder Length="8" Function="Wheel" LeftScale="1.0" RightScale="1.0" Magnification="1.0"/>
        <BatterySencer Length="2" Function="Battery"/>
        <IncrementalSteeringCoder Length="2" Function="Steer" SteeringZero="0.0" Magnification="1.0"/>
        <Gyroscope Length="4" Function="Gyro" Magnification="1.0"/>
        <Accelerometer Length="2" Function="Accel" Zero="0.0" Magnification="1.0"/>
        <HeightCoder Length="4" Function="Height" Zero="0.0" Magnification="1.0"/>
        <DataIndex Length="4" Function="Index"/>
        <DataCRC Length="2" Function="CRC"/>
        <DataTail Length="1" Function="Tail"/>
    </Sensors>
</Config>"""
    
    # Write config files
    with open("sample_actuators.config", "w") as f:
        f.write(actuator_config)
    
    with open("sample_sensors.config", "w") as f:
        f.write(sensor_config)
    
    print("Sample configuration files created:")
    print("- sample_actuators.config")
    print("- sample_sensors.config")


def main():
    """Main test function"""
    print("Python Parser Example Usage")
    print("=" * 40)
    
    # Create sample configs
    create_sample_config()
    
    # Test with sample configs
    print("\n=== Testing with Sample Configs ===")
    
    # Test actuator decoding
    decoder = InputDecoder()
    ret = decoder.load_config("sample_actuators.config")
    if ret == 0:
        print("Actuator config loaded successfully")
        test_actuator_decoding()
    else:
        print(f"Failed to load actuator config: {ret}")
    
    # Test sensor encoding
    encoder = OutputEncoder()
    ret = encoder.load_config("sample_sensors.config")
    if ret == 0:
        print("Sensor config loaded successfully")
        test_sensor_encoding()
    else:
        print(f"Failed to load sensor config: {ret}")
    
    # Test switch operations
    test_switch_operations()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
