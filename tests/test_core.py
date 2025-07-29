"""Tests for core liquid neural network functionality."""

import pytest
import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig


class TestLiquidConfig:
    """Test LiquidConfig class."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = LiquidConfig(
            input_dim=4,
            hidden_dim=8,
            output_dim=2
        )
        assert config.input_dim == 4
        assert config.hidden_dim == 8
        assert config.output_dim == 2

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            LiquidConfig(input_dim=0, hidden_dim=8, output_dim=2)
        
        with pytest.raises(ValueError):
            LiquidConfig(input_dim=4, hidden_dim=0, output_dim=2)


class TestLiquidNN:
    """Test LiquidNN class."""

    def test_model_creation(self, basic_config):
        """Test model creation with config."""
        model = LiquidNN(basic_config)
        assert model.config == basic_config

    def test_model_initialization(self, basic_config, rng_key, sample_input):
        """Test model parameter initialization."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_input)
        
        # Check parameter structure
        assert "params" in params
        assert isinstance(params["params"], dict)

    def test_forward_pass(self, basic_config, rng_key, sample_input):
        """Test forward pass execution."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_input)
        output = model.apply(params, sample_input)
        
        # Check output shape and validity
        assert output.shape == (1, basic_config.output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_batch_processing(self, basic_config, rng_key, sample_batch):
        """Test batch processing."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_batch[:1])  # Init with single sample
        output = model.apply(params, sample_batch)
        
        # Check batch output shape
        assert output.shape == (sample_batch.shape[0], basic_config.output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_deterministic_output(self, basic_config, rng_key, sample_input):
        """Test that identical inputs produce identical outputs."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_input)
        
        output1 = model.apply(params, sample_input)
        output2 = model.apply(params, sample_input)
        
        assert jnp.allclose(output1, output2, atol=1e-6)