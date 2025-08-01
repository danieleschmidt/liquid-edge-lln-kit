# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Liquid Edge LLN Kit project.

## ADR Format

Each ADR follows the standard format:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue that we're seeing that is motivating this decision?
- **Decision**: What is the change that we're proposing or have agreed to implement?
- **Consequences**: What becomes easier or more difficult to do and any risks introduced by this change?

## ADR List

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [001](001-jax-backend-choice.md) | JAX as Primary ML Backend | Accepted | 2025-01-15 |
| [002](002-cmsis-nn-deployment.md) | CMSIS-NN for ARM Deployment | Accepted | 2025-01-15 |
| [003](003-energy-first-design.md) | Energy-First Design Philosophy | Accepted | 2025-01-15 |

## Creating New ADRs

1. Copy the [template](template.md)
2. Number sequentially (001, 002, etc.)
3. Use kebab-case for filename
4. Update this README with the new entry