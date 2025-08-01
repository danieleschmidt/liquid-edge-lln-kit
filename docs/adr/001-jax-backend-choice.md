# ADR-001: JAX as Primary ML Backend

## Status

Accepted

## Context

The Liquid Edge LLN Kit requires a high-performance ML framework that supports:
- Functional programming paradigms for neural ODE implementations
- Just-in-time compilation for edge deployment optimization
- Hardware-agnostic computation graphs
- Research-friendly APIs for liquid neural network experimentation

## Decision

We will use JAX as the primary machine learning backend, with Flax for neural network layers and Optax for optimization.

## Consequences

**Positive:**
- Excellent performance through XLA compilation
- Functional programming model matches liquid neural network mathematics
- Strong ecosystem (Flax, Optax, JAX-MD)
- Easy gradient computation and differentiation
- Hardware-agnostic deployment (CPU/GPU/TPU)

**Negative:**
- Smaller community compared to PyTorch/TensorFlow
- Steeper learning curve for developers unfamiliar with functional programming
- Limited pre-trained model availability
- Potential debugging complexity with transformed functions

## Alternatives Considered

1. **PyTorch**: Larger ecosystem but imperative paradigm less suitable for ODEs
2. **TensorFlow**: More deployment options but graph execution model less flexible
3. **Custom C++**: Maximum performance but development velocity too slow

## Implementation Notes

- Use Flax for all neural network modules
- Leverage JAX transformations (jit, vmap, grad) extensively
- Implement custom liquid cell operations as pure functions
- Use Optax for energy-aware optimization schedules

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Liquid Neural Networks Paper](https://arxiv.org/abs/2006.04439)
- [Flax Examples](https://github.com/google/flax/tree/main/examples)