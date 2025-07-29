# Deployment Guide

This guide covers deploying Liquid Neural Networks to various edge devices and platforms.

## 🎯 Deployment Targets

### Microcontrollers
- **ARM Cortex-M4/M7**: STM32H7, STM32F4, STM32L4 series
- **ESP32**: ESP32, ESP32-S3, ESP32-C3 with ESP-NN acceleration
- **Nordic**: nRF52840, nRF5340 with ARM optimizations
- **Teensy**: 4.0, 4.1 for high-performance applications

### Single Board Computers  
- **Raspberry Pi**: 4B, 5, Zero 2W with ARM NEON optimizations
- **Jetson**: Nano, Orin Nano with CUDA acceleration
- **Rock Pi**: 4, 5 series with Mali GPU support

### Development Boards
- **Arduino**: Portenta H7, Nano 33 BLE Sense
- **Adafruit**: Feather M4, PyPortal
- **SparkFun**: Thing Plus, MicroMod

## ⚙️ Platform-Specific Setup

### STM32 Deployment

```python
from liquid_edge.deploy import STM32Deployer

deployer = STM32Deployer(
    chip="STM32H743VIT6",
    clock_speed=480_000_000,  # 480MHz
    memory_profile="optimized",
    use_cmsis_nn=True
)

# Configure for your board
deployer.configure_peripherals({
    "uart": {"baud": 115200, "pins": ["PA9", "PA10"]},
    "i2c": {"speed": 400000, "pins": ["PB8", "PB9"]},
    "spi": {"speed": 10000000, "pins": ["PA5", "PA6", "PA7"]}
})

# Deploy model
deployer.deploy(
    model_path="trained_model.liquid",
    output_dir="stm32_firmware/",
    optimization_level="O3"
)
```

### ESP32 Deployment

```python
from liquid_edge.deploy import ESP32Deployer

deployer = ESP32Deployer(
    chip="ESP32S3",
    flash_size="8MB",
    psram=True,
    use_esp_nn=True
)

# Configure WiFi for OTA updates
deployer.configure_wifi({
    "ssid": "your_network",
    "password": "your_password",
    "enable_ota": True
})

# Deploy with monitoring
deployer.deploy(
    model_path="trained_model.liquid",
    include_telemetry=True,
    energy_monitoring=True
)
```

### Raspberry Pi Deployment

```python
from liquid_edge.deploy import RaspberryPiDeployer

deployer = RaspberryPiDeployer(
    model="pi4b",
    use_neon=True,  # ARM NEON optimizations
    use_gpu=False,  # VideoCore GPU (limited)
    memory_split=64  # GPU memory split
)

# Deploy as systemd service
deployer.deploy_service(
    model_path="trained_model.liquid",
    service_name="liquid-inference",
    auto_start=True
)
```

## 🔧 Build System Integration

### PlatformIO Configuration

```ini
; platformio.ini
[env:stm32h743vi]
platform = ststm32
board = nucleo_h743zi
framework = cmsis

build_flags = 
    -DLIQUID_NN_OPTIMIZED
    -DARM_MATH_CM7
    -DUSE_HAL_DRIVER
    -O3
    -ffast-math

lib_deps = 
    ARM-software/CMSIS@^5.9.0
    liquid-edge/cmsis-nn-liquid@^1.0.0

monitor_speed = 115200
```

### ESP-IDF Component

```cmake
# CMakeLists.txt
idf_component_register(
    SRCS "liquid_nn.c" "esp32_inference.c"
    INCLUDE_DIRS "include"
    REQUIRES esp-nn spi_flash nvs_flash
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -ffast-math
    -DESP_NN_OPTIMIZED
)
```

### Arduino Library

```cpp
// library.properties
name=LiquidEdgeNN
version=0.1.0
author=Liquid Edge Team
maintainer=liquid-edge@example.com
sentence=Liquid Neural Networks for Arduino
paragraph=Efficient LNN inference on microcontrollers
category=Machine Learning
url=https://github.com/liquid-edge/arduino-lib
architectures=esp32,stm32,sam
depends=ArduinoJson,WiFi
```

## 🚀 Continuous Deployment

### GitHub Actions Integration

```yaml
# .github/workflows/deploy-edge.yml
name: Edge Deployment
on:
  release:
    types: [published]

jobs:
  build-firmware:
    strategy:
      matrix:
        target: [stm32h7, esp32s3, nrf52840]
    
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup PlatformIO
        run: pip install platformio
        
      - name: Build firmware
        run: |
          liquid-lln deploy --target ${{ matrix.target }} \
                           --model models/production.liquid \
                           --output firmware/${{ matrix.target }}/
          
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: firmware-${{ matrix.target }}
          path: firmware/${{ matrix.target }}/
```

### OTA Update System

```python
from liquid_edge.ota import OTAManager

# Configure OTA updates
ota = OTAManager(
    device_id="robot_001",
    server_url="https://ota.liquid-edge.com",
    cert_file="ca-cert.pem"
)

# Check for updates
if ota.check_for_updates():
    print(f"New version available: {ota.latest_version}")
    
    # Download and verify
    if ota.download_update():
        # Apply update safely
        ota.apply_update(backup_current=True)
        ota.reboot()
```

## 📊 Performance Monitoring

### Real-time Metrics

```python
from liquid_edge.monitoring import EdgeMetrics

metrics = EdgeMetrics(
    device_id="edge_device_001",
    upload_interval=60,  # seconds
    enable_telemetry=True
)

# Monitor during inference
with metrics.measure_inference():
    output = model.infer(sensor_data)
    
# Collect system metrics
metrics.log_system_stats()
metrics.log_energy_consumption()
metrics.upload_batch()
```

### Dashboard Integration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'liquid-edge-devices'
    static_configs:
      - targets: ['robot-001:8080', 'robot-002:8080']
    scrape_interval: 5s
    metrics_path: /metrics
```

## 🔒 Security Best Practices

### Model Encryption

```python
from liquid_edge.security import ModelEncryption

# Encrypt model for deployment
encryptor = ModelEncryption(
    key_file="device_key.pem",
    algorithm="AES-256-GCM"
)

encrypted_model = encryptor.encrypt_model(
    model_path="trained_model.liquid",
    output_path="encrypted_model.bin"
)

# Deploy with secure boot
deployer.deploy_secure(
    encrypted_model=encrypted_model,
    enable_secure_boot=True,
    code_signing_key="signing_key.pem"
)
```

### Certificate Management

```bash
# Generate device certificates
liquid-lln security generate-certs \
    --device-id robot_001 \
    --ca-cert ca.pem \
    --ca-key ca-key.pem \
    --output-dir certs/

# Provision device
liquid-lln security provision \
    --device /dev/ttyUSB0 \
    --cert certs/robot_001.pem \
    --key certs/robot_001-key.pem
```

## 🧪 Testing Deployment

### Hardware-in-Loop Testing

```python
from liquid_edge.testing import HILTestSuite

# Configure test hardware
test_suite = HILTestSuite([
    {"device": "stm32h743", "port": "/dev/ttyUSB0"},
    {"device": "esp32s3", "port": "/dev/ttyUSB1"},
    {"device": "nrf52840", "port": "/dev/ttyACM0"}
])

# Run deployment tests
results = test_suite.run_deployment_tests(
    firmware_dir="firmware/",
    test_cases=[
        "basic_inference",
        "energy_consumption", 
        "real_time_performance",
        "error_recovery"
    ]
)

# Generate test report
test_suite.generate_report(results, "deployment_test_report.html")
```

### Automated Validation

```bash
# CI/CD validation pipeline
liquid-lln validate --firmware firmware/ \
                   --hardware-config hardware.yml \
                   --performance-targets performance.json \
                   --security-scan \
                   --generate-report
```

## 📚 Platform-Specific Guides

- [STM32 Detailed Guide](platforms/stm32.md)
- [ESP32 Setup Guide](platforms/esp32.md) 
- [Raspberry Pi Guide](platforms/raspberry-pi.md)
- [Arduino Integration](platforms/arduino.md)
- [ROS2 Robot Deployment](platforms/ros2.md)

## 🆘 Troubleshooting

### Common Issues

**Memory Overflow**
```bash
# Reduce model size
liquid-lln optimize --target esp32 --memory-limit 256KB model.liquid

# Enable memory profiling
liquid-lln deploy --memory-profile --target stm32h7 model.liquid
```

**Performance Issues**
```bash
# Profile execution
liquid-lln profile --device hardware --iterations 1000 model.liquid

# Optimize for target
liquid-lln optimize --target cortex-m7 --clock 480MHz model.liquid
```

**Communication Problems**
```bash
# Test device connection
liquid-lln device test --port /dev/ttyUSB0

# Flash bootloader
liquid-lln device flash-bootloader --port /dev/ttyUSB0 --chip esp32s3
```

## 📞 Support

- Hardware-specific issues: [Platform Forums](https://forum.liquid-edge.com)
- Deployment bugs: [GitHub Issues](https://github.com/liquid-edge/issues)
- Integration help: [Discord Community](https://discord.gg/liquid-edge)