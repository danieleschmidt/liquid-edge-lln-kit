# ADR-002: CMSIS-NN for ARM Deployment

## Status

Accepted

## Context

Edge robotics applications require deployment on ARM Cortex-M microcontrollers with severe memory and compute constraints. We need optimized neural network inference libraries that can run liquid neural networks efficiently on these platforms.

## Decision

We will use ARM's CMSIS-NN library as the primary inference engine for ARM Cortex-M deployments, with custom liquid state update kernels.

## Consequences

**Positive:**
- Highly optimized kernels for ARM Cortex-M processors
- INT8 quantization support with minimal accuracy loss
- Memory-efficient implementations
- Industry standard with broad hardware support
- Integration with existing ARM development tools

**Negative:**
- ARM-specific, limits cross-platform portability
- Requires custom kernel development for liquid dynamics
- INT8 quantization may impact liquid network expressiveness
- Additional complexity in build system

## Alternatives Considered

1. **TensorFlow Lite Micro**: More portable but less optimized for ARM
2. **ONNX Runtime**: Better cross-platform but larger memory footprint
3. **Custom C kernels**: Full control but significant development overhead
4. **Arm NN**: Too heavyweight for microcontrollers

## Implementation Notes

- Generate CMSIS-NN compatible C code from JAX models
- Implement custom liquid state update kernels in ARM assembly
- Use INT8 quantization with calibration datasets
- Integrate with ARM Keil MDK and GCC toolchains
- Provide fallback implementations for non-ARM platforms

## References

- [CMSIS-NN Documentation](https://arm-software.github.io/CMSIS_5/NN/html/index.html)
- [ARM Cortex-M Performance Guidelines](https://developer.arm.com/documentation/dai0425/latest)
- [Quantization Best Practices](https://www.tensorflow.org/lite/performance/post_training_quantization)