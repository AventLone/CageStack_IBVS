# Python Parser

This is a Python conversion of the C++ parser module for Webots control dataset. The parser handles encoding and decoding of actuator and sensor data based on XML configuration files.

## Features

- **InputDecoder**: Parses actuator data from binary frames based on XML configuration
- **OutputEncoder**: Creates sensor data packages based on XML configuration
- **Modular Design**: Separate model classes for different actuator and sensor types
- **Type Safety**: Proper handling of signed/unsigned and 16/32-bit data types
- **CRC Support**: Automatic CRC calculation for data integrity

## Structure

```
python_parser/
├── __init__.py          # Module initialization
├── parser.py            # Main InputDecoder and OutputEncoder classes
├── models.py            # Actuator and sensor model classes
├── directors.py         # Director classes for model management
├── utils.py             # Utility functions for byte manipulation
├── example_usage.py     # Example usage and testing
└── README.md           # This file
```

## Classes

### Main Parser Classes

- **InputDecoder**: Decodes actuator data from binary frames
- **OutputEncoder**: Encodes sensor data into binary frames
- **Package**: Data package structure
- **UpdateValue**: Update value structure for sensor data
- **SpecialParam**: Special parameter structure

### Actuator Models

- **MoveDevice**: Move device with forward/backward polynomial mapping
- **SteeringDevice**: Steering device with polynomial mapping
- **ForkDevice**: Fork device with positive/negative polynomial mapping
- **LiftDevice**: Lift device model
- **SwitchActuator**: Switch actuator for bit manipulation
- **ADataIndex**: Data index model
- **ADataIgnore**: Data ignore model
- **MCUDataIndexReturn**: MCU data index return model
- **ValveControleDevice**: Valve control device model
- **SerialDataActuator**: Serial data actuator model
- **GPIOSwitchActuator**: GPIO switch actuator model
- **CanToWifiActuator**: CAN to WiFi actuator model

### Sensor Models

- **WheelCoder**: Wheel coder for left/right wheel encoding
- **BatterySencer**: Battery sensor model
- **IncrementalSteeringCoder**: Incremental steering coder model
- **Gyroscope**: Gyroscope model
- **ElePerceptionCameraDistance**: Electronic perception camera distance model
- **SDataIndex**: Sensor data index model
- **ForkDisplacementSencer**: Fork displacement sensor model
- **HeightCoder**: Height coder model
- **HolzerCoder**: Holzer coder model
- **DataIndexReturn**: Data index return model
- **NullSencer**: Null sensor model
- **ErrorCode**: Error code model
- **RPMSensor**: RPM sensor model
- **VelocityControlLevel**: Velocity control level model
- **SwitchSencer**: Switch sensor for bit manipulation
- **Accelerometer**: Accelerometer model
- **AngularVelocitySensor**: Angular velocity sensor model
- **HydraulicPressureSensor**: Hydraulic pressure sensor model
- **DataCRC**: Data CRC model
- **DataTail**: Data tail model

## Usage

### Basic Usage

```python
from python_parser import InputDecoder, OutputEncoder, Package

# Create decoder
decoder = InputDecoder()
decoder.load_config("actuators.config")

# Decode data
data = bytes([0xAA, 0x00, 0x01, ...])  # Your binary data
package = Package(buf=data, len=len(data))
decoder.decode_package(package)

# Get values
output = []
decoder.get_value("MoveDevice", output)
print(f"Move device value: {output[0]}")

# Create encoder
encoder = OutputEncoder()
encoder.load_config("sensors.config")

# Update values
encoder.update_value("WheelCoder", 2, "", 10.0, 10.0)
encoder.update_value("Gyroscope", 1, "", 3.14)

# Encode package
package = encoder.encode_package()
print(f"Encoded data: {package.buf.hex()}")
```

### Configuration Files

The parser uses XML configuration files to define the structure of actuator and sensor data. Example:

```xml
<?xml version="1.0"?>
<Config>
    <Actuators>
        <DataHeader Length="1" Function="Header"/>
        <MoveDevice Length="2" Function="Move" ForwardPoly="1.0,0.0" BackwardPoly="1.0,0.0" MaxControl="100.0" MinControl="-100.0"/>
        <SteeringDevice Length="2" Function="Steer" Poly="1.0,0.0"/>
        <DataTail Length="1" Function="Tail"/>
    </Actuators>
</Config>
```

## Key Differences from C++

1. **Memory Management**: Python handles memory management automatically
2. **Type System**: Python uses dynamic typing with type hints for clarity
3. **Error Handling**: Python uses exceptions and return codes
4. **Data Structures**: Uses Python dataclasses and built-in collections
5. **XML Parsing**: Uses Python's built-in xml.etree.ElementTree
6. **Byte Operations**: Uses Python's struct module for binary data handling

## Dependencies

- Python 3.7+
- Standard library only (no external dependencies)

## Testing

Run the example usage script to test the parser:

```bash
python example_usage.py
```

This will create sample configuration files and demonstrate the parser functionality.

## Conversion Notes

This Python version maintains the same API structure as the original C++ code while adapting to Python idioms and best practices. The core functionality remains the same, ensuring compatibility with existing configuration files and data formats.
