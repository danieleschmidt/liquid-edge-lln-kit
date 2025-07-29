# Examples

This directory contains example implementations and tutorials for the Liquid Edge LLN Kit.

## Structure

```
examples/
├── basic/              # Basic liquid network examples
├── robotics/           # Robot control applications
├── sensors/            # Sensor fusion examples
├── deployment/         # MCU deployment examples
└── benchmarks/         # Performance benchmarking
```

## Quick Start Examples

### 1. Basic Liquid Network
```python
# examples/basic/simple_liquid.py
from liquid_edge import LiquidNN, LiquidConfig

config = LiquidConfig(input_dim=4, hidden_dim=8, output_dim=2)
model = LiquidNN(config)
# ... training and inference code
```

### 2. Robot Line Following
```python
# examples/robotics/line_follower.py
from liquid_edge.robotics import LineFollowerController

controller = LineFollowerController(sensor_count=5)
# ... robot control implementation
```

### 3. MCU Deployment
```python
# examples/deployment/stm32_deploy.py
from liquid_edge.deploy import MCUDeployer, TargetDevice

deployer = MCUDeployer(target=TargetDevice.STM32H743)
deployer.export_model(model, "firmware/")
```

## Running Examples

Each example includes:
- **README.md**: Description and requirements
- **requirements.txt**: Additional dependencies
- **run.py**: Main execution script
- **config.yaml**: Configuration parameters

To run an example:
```bash
cd examples/basic/simple_liquid/
pip install -r requirements.txt
python run.py
```

## Contributing Examples

We welcome example contributions! Please:
1. Include complete working code
2. Add clear documentation
3. Specify hardware requirements
4. Include expected outputs
5. Test on target platforms

## Hardware Requirements

Different examples require different hardware:
- **Basic examples**: Any system with Python 3.10+
- **Robot examples**: Specific robot platforms (documented per example)
- **MCU examples**: Development boards and toolchains
- **ROS 2 examples**: ROS 2 Humble installation

## Getting Help

- Check example-specific README files
- Review the main documentation in `/docs`
- Open issues for example-specific problems
- Join the community Discord for real-time help