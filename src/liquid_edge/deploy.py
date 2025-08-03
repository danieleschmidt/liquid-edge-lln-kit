"""Hardware deployment tools for liquid neural networks."""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import jax.numpy as jnp
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path


class TargetDevice(Enum):
    """Supported MCU targets."""
    STM32H743 = "stm32h743"
    STM32F767 = "stm32f767" 
    ESP32_S3 = "esp32s3"
    NRF52840 = "nrf52840"
    TEENSY41 = "teensy41"
    CUSTOM = "custom"


@dataclass
class DeploymentConfig:
    """Configuration for MCU deployment."""
    target: TargetDevice
    optimization_level: str = "O3"
    quantization: str = "int8"
    memory_limit_kb: int = 256
    clock_speed_mhz: int = 400
    include_cmsis_nn: bool = True
    use_fixed_point: bool = True
    enable_profiling: bool = False
    flash_size_kb: int = 2048
    ram_size_kb: int = 512


class MCUDeployer:
    """Deploy liquid neural networks to microcontrollers."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.target_specs = self._get_target_specs()
        
    def _get_target_specs(self) -> Dict[str, Any]:
        """Get hardware specifications for target device."""
        specs = {
            TargetDevice.STM32H743: {
                "arch": "arm_cortex_m7",
                "fpu": "fpv5-d16",
                "compiler": "arm-none-eabi-gcc",
                "linker_script": "stm32h743.ld",
                "defines": ["STM32H743xx", "ARM_MATH_CM7"],
                "libs": ["m", "c", "nosys"],
                "cmsis_variant": "cortex_m7"
            },
            TargetDevice.ESP32_S3: {
                "arch": "xtensa_esp32s3", 
                "compiler": "xtensa-esp32s3-elf-gcc",
                "framework": "esp-idf",
                "defines": ["ESP32S3", "CONFIG_FREERTOS_HZ=1000"],
                "libs": ["esp_nn", "esp_dsp"],
                "optimization_libs": ["esp-nn"]
            },
            TargetDevice.NRF52840: {
                "arch": "arm_cortex_m4",
                "fpu": "fpv4-sp-d16", 
                "compiler": "arm-none-eabi-gcc",
                "defines": ["NRF52840_XXAA", "ARM_MATH_CM4"],
                "libs": ["m", "c", "nosys"],
                "cmsis_variant": "cortex_m4"
            }
        }
        return specs.get(self.config.target, {})
    
    def export_model(self, 
                     model, 
                     params: Dict[str, Any], 
                     output_dir: str,
                     model_name: str = "liquid_model") -> str:
        """Export trained model to optimized C code."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model parameters
        model_params = self._extract_parameters(params)
        
        # Generate C header file
        header_path = os.path.join(output_dir, f"{model_name}.h")
        self._generate_header(model_params, header_path, model_name)
        
        # Generate C implementation  
        source_path = os.path.join(output_dir, f"{model_name}.c")
        self._generate_source(model_params, source_path, model_name)
        
        # Generate CMakeLists.txt or Makefile
        build_file = self._generate_build_config(output_dir, model_name)
        
        print(f"Model exported to: {output_dir}")
        print(f"Header: {header_path}")
        print(f"Source: {source_path}")
        print(f"Build config: {build_file}")
        
        return output_dir
    
    def _extract_parameters(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract and quantize model parameters."""
        extracted = {}
        
        def process_layer(layer_params, prefix=""):
            for key, value in layer_params.items():
                if isinstance(value, dict):
                    process_layer(value, f"{prefix}{key}_")
                else:
                    # Convert JAX array to numpy
                    if hasattr(value, 'shape'):
                        param_name = f"{prefix}{key}"
                        param_array = np.array(value)
                        
                        # Apply quantization
                        if self.config.quantization == "int8":
                            quantized = self._quantize_int8(param_array)
                            extracted[param_name] = quantized
                        elif self.config.quantization == "int16":
                            quantized = self._quantize_int16(param_array)
                            extracted[param_name] = quantized
                        else:
                            extracted[param_name] = param_array
        
        process_layer(params)
        return extracted
    
    def _quantize_int8(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Quantize weights to INT8 with scale and zero-point."""
        w_min, w_max = weights.min(), weights.max()
        
        # Calculate scale and zero-point
        scale = (w_max - w_min) / 255.0
        zero_point = int(-w_min / scale)
        zero_point = np.clip(zero_point, 0, 255)
        
        # Quantize
        quantized = np.round(weights / scale + zero_point)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return quantized, scale, zero_point
    
    def _quantize_int16(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize weights to INT16 with scale factor."""
        scale = np.max(np.abs(weights)) / 32767.0
        quantized = np.round(weights / scale).astype(np.int16)
        return quantized, scale
    
    def _generate_header(self, params: Dict[str, np.ndarray], 
                        header_path: str, model_name: str):
        """Generate C header file with model parameters."""
        with open(header_path, 'w') as f:
            f.write(f"""#ifndef {model_name.upper()}_H
#define {model_name.upper()}_H

#ifdef __cplusplus
extern "C" {{
#endif

#include <stdint.h>
#include <stddef.h>

""")
            
            if self.config.include_cmsis_nn:
                f.write('#include "arm_nnfunctions.h"\n\n')
            
            # Model dimensions
            f.write("// Model Configuration\n")
            f.write(f"#define INPUT_DIM {self._get_input_dim(params)}\n")
            f.write(f"#define HIDDEN_DIM {self._get_hidden_dim(params)}\n") 
            f.write(f"#define OUTPUT_DIM {self._get_output_dim(params)}\n")
            f.write(f"#define QUANTIZATION_BITS {8 if self.config.quantization == 'int8' else 16}\n\n")
            
            # Parameter declarations
            f.write("// Model Parameters\n")
            for name, param in params.items():
                if isinstance(param, tuple):  # Quantized parameter
                    weights, scale, zero_point = param
                    f.write(f"extern const uint8_t {name}_weights[{weights.size}];\n")
                    f.write(f"extern const float {name}_scale;\n")
                    f.write(f"extern const int {name}_zero_point;\n")
                else:
                    dtype = "int8_t" if self.config.quantization == "int8" else "int16_t"
                    f.write(f"extern const {dtype} {name}[{param.size}];\n")
            
            f.write("\n// State Structure\n")
            f.write(f"""typedef struct {{
    float hidden_state[HIDDEN_DIM];
    float tau[HIDDEN_DIM];
    int32_t step_count;
}} {model_name}_state_t;

""")
            
            # Function declarations
            f.write("// Function Declarations\n")
            f.write(f"void {model_name}_init({model_name}_state_t* state);\n")
            f.write(f"void {model_name}_inference(const float* input, float* output, {model_name}_state_t* state);\n")
            
            if self.config.quantization == "int8":
                f.write(f"void {model_name}_inference_q8(const int8_t* input, int8_t* output, {model_name}_state_t* state);\n")
            
            if self.config.enable_profiling:
                f.write(f"uint32_t {model_name}_get_cycles(void);\n")
                f.write(f"float {model_name}_get_energy_mw(void);\n")
            
            f.write(f"""
#ifdef __cplusplus
}}
#endif

#endif // {model_name.upper()}_H
""")
    
    def _generate_source(self, params: Dict[str, np.ndarray], 
                        source_path: str, model_name: str):
        """Generate optimized C implementation."""
        with open(source_path, 'w') as f:
            f.write(f'#include "{model_name}.h"\n')
            f.write('#include <math.h>\n')
            f.write('#include <string.h>\n\n')
            
            if self.config.enable_profiling:
                f.write('#include "system_profiler.h"\n\n')
            
            # Parameter definitions
            f.write("// Model Parameters\n")
            for name, param in params.items():
                if isinstance(param, tuple):  # Quantized
                    weights, scale, zero_point = param
                    f.write(f"const uint8_t {name}_weights[{weights.size}] = {{\n")
                    self._write_array_data(f, weights.flatten())
                    f.write("};\n")
                    f.write(f"const float {name}_scale = {scale:.8f}f;\n")
                    f.write(f"const int {name}_zero_point = {zero_point};\n\n")
                else:
                    dtype = "int8_t" if self.config.quantization == "int8" else "int16_t"
                    f.write(f"const {dtype} {name}[{param.size}] = {{\n")
                    self._write_array_data(f, param.flatten())
                    f.write("};\n\n")
            
            # Utility functions
            f.write(self._generate_utility_functions(model_name))
            
            # Main inference function
            f.write(self._generate_inference_function(model_name, params))
            
            # Quantized inference if enabled
            if self.config.quantization == "int8":
                f.write(self._generate_quantized_inference(model_name, params))
    
    def _generate_inference_function(self, model_name: str, params: Dict[str, np.ndarray]) -> str:
        """Generate main inference function."""
        return f"""
void {model_name}_inference(const float* input, float* output, {model_name}_state_t* state) {{
    {"uint32_t start_cycles = get_cycle_count();" if self.config.enable_profiling else ""}
    
    // Temporary buffers
    float input_proj[HIDDEN_DIM];
    float recurrent_proj[HIDDEN_DIM]; 
    float activation[HIDDEN_DIM];
    
    // Input projection: input_proj = input * W_in
    matrix_vector_multiply_f32(input, input_projection_kernel, input_proj, INPUT_DIM, HIDDEN_DIM);
    vector_add_f32(input_proj, input_projection_bias, input_proj, HIDDEN_DIM);
    
    // Recurrent projection: recurrent_proj = hidden * W_rec  
    matrix_vector_multiply_f32(state->hidden_state, recurrent_projection_kernel, recurrent_proj, HIDDEN_DIM, HIDDEN_DIM);
    vector_add_f32(recurrent_proj, recurrent_projection_bias, recurrent_proj, HIDDEN_DIM);
    
    // Activation: tanh(input_proj + recurrent_proj)
    vector_add_f32(input_proj, recurrent_proj, activation, HIDDEN_DIM);
    vector_tanh_f32(activation, activation, HIDDEN_DIM);
    
    // Update adaptive time constants
    update_time_constants(state->hidden_state, state->tau);
    
    // Liquid dynamics: h_new = h + dt * (-h + activation) / tau
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        float dx_dt = (-state->hidden_state[i] + activation[i]) / state->tau[i];
        state->hidden_state[i] += DT * dx_dt;
    }}
    
    // Output projection: output = hidden * W_out + b_out
    matrix_vector_multiply_f32(state->hidden_state, output_layer_kernel, output, HIDDEN_DIM, OUTPUT_DIM);
    vector_add_f32(output, output_layer_bias, output, OUTPUT_DIM);
    
    state->step_count++;
    
    {"uint32_t end_cycles = get_cycle_count();" if self.config.enable_profiling else ""}
    {"update_energy_estimate(end_cycles - start_cycles);" if self.config.enable_profiling else ""}
}}
"""
    
    def _generate_utility_functions(self, model_name: str) -> str:
        """Generate optimized utility functions."""
        return f"""
// Optimized Math Functions
static inline void matrix_vector_multiply_f32(const float* matrix, const float* vector, 
                                            float* result, int rows, int cols) {{
    for (int i = 0; i < cols; i++) {{
        result[i] = 0.0f;
        for (int j = 0; j < rows; j++) {{
            result[i] += matrix[i * rows + j] * vector[j];
        }}
    }}
}}

static inline void vector_add_f32(const float* a, const float* b, float* result, int size) {{
    for (int i = 0; i < size; i++) {{
        result[i] = a[i] + b[i];
    }}
}}

static inline void vector_tanh_f32(const float* input, float* output, int size) {{
    for (int i = 0; i < size; i++) {{
        // Fast tanh approximation
        float x = input[i];
        if (x > 2.0f) output[i] = 1.0f;
        else if (x < -2.0f) output[i] = -1.0f;
        else {{
            float x2 = x * x;
            output[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
        }}
    }}
}}

static void update_time_constants(const float* hidden, float* tau) {{
    // Update adaptive time constants
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        float tau_raw = hidden[i] * 0.1f;  // Simplified projection
        tau[i] = TAU_MIN + (TAU_MAX - TAU_MIN) * (1.0f / (1.0f + expf(-tau_raw)));
    }}
}}

void {model_name}_init({model_name}_state_t* state) {{
    // Initialize hidden state to zero
    memset(state->hidden_state, 0, sizeof(state->hidden_state));
    
    // Initialize time constants
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        state->tau[i] = (TAU_MIN + TAU_MAX) / 2.0f;
    }}
    
    state->step_count = 0;
}}

#define TAU_MIN 10.0f
#define TAU_MAX 100.0f
#define DT 0.1f
"""
    
    def _generate_quantized_inference(self, model_name: str, params: Dict[str, np.ndarray]) -> str:
        """Generate INT8 quantized inference function."""
        return f"""
void {model_name}_inference_q8(const int8_t* input, int8_t* output, {model_name}_state_t* state) {{
    // INT8 quantized inference implementation
    // This would use CMSIS-NN functions for optimal performance
    
    if (state->step_count == 0) {{
        // First inference - initialize quantized state
        {model_name}_init(state);
    }}
    
    // Convert input to float (dequantize)
    float input_f32[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) {{
        input_f32[i] = (float)(input[i] - INPUT_ZERO_POINT) * INPUT_SCALE;
    }}
    
    // Run float inference
    float output_f32[OUTPUT_DIM];
    {model_name}_inference(input_f32, output_f32, state);
    
    // Convert output back to INT8 (quantize)
    for (int i = 0; i < OUTPUT_DIM; i++) {{
        int32_t quantized = (int32_t)(output_f32[i] / OUTPUT_SCALE + OUTPUT_ZERO_POINT);
        output[i] = (int8_t)CLAMP(quantized, -128, 127);
    }}
}}

#define INPUT_SCALE 0.01f
#define INPUT_ZERO_POINT 128
#define OUTPUT_SCALE 0.01f  
#define OUTPUT_ZERO_POINT 128
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
"""
    
    def _write_array_data(self, f, data: np.ndarray, items_per_line: int = 12):
        """Write array data in readable C format."""
        for i, val in enumerate(data):
            if i % items_per_line == 0:
                f.write("  ")
            
            if isinstance(val, (np.integer, int)):
                f.write(f"{val}")
            else:
                f.write(f"{val:.6f}f")
                
            if i < len(data) - 1:
                f.write(", ")
            
            if (i + 1) % items_per_line == 0 or i == len(data) - 1:
                f.write("\n")
    
    def _get_input_dim(self, params: Dict[str, np.ndarray]) -> int:
        """Extract input dimension from parameters."""
        # Look for input projection weights
        for name, param in params.items():
            if "input_projection" in name and "kernel" in name:
                if isinstance(param, tuple):
                    return param[0].shape[0]
                return param.shape[0]
        return 4  # Default
    
    def _get_hidden_dim(self, params: Dict[str, np.ndarray]) -> int:
        """Extract hidden dimension from parameters."""
        for name, param in params.items():
            if "input_projection" in name and "kernel" in name:
                if isinstance(param, tuple):
                    return param[0].shape[1]
                return param.shape[1]
        return 8  # Default
    
    def _get_output_dim(self, params: Dict[str, np.ndarray]) -> int:
        """Extract output dimension from parameters."""
        for name, param in params.items():
            if "output_layer" in name and "kernel" in name:
                if isinstance(param, tuple):
                    return param[0].shape[1]
                return param.shape[1]
        return 2  # Default
    
    def _generate_build_config(self, output_dir: str, model_name: str) -> str:
        """Generate build configuration file."""
        if self.config.target == TargetDevice.ESP32_S3:
            return self._generate_esp_idf_config(output_dir, model_name)
        else:
            return self._generate_makefile(output_dir, model_name)
    
    def _generate_makefile(self, output_dir: str, model_name: str) -> str:
        """Generate Makefile for ARM targets."""
        makefile_path = os.path.join(output_dir, "Makefile")
        
        with open(makefile_path, 'w') as f:
            f.write(f"""# Makefile for {model_name} - {self.config.target.value}

# Compiler Configuration
CC = {self.target_specs.get('compiler', 'arm-none-eabi-gcc')}
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

# Target Configuration  
TARGET = {model_name}
MCU = {self.target_specs.get('arch', 'cortex-m7')}
FPU = {self.target_specs.get('fpu', 'fpv5-d16')}

# Compilation Flags
CFLAGS = -mcpu=$(MCU) -mthumb -mfpu=$(FPU) -mfloat-abi=hard
CFLAGS += -{self.config.optimization_level} -Wall -Wextra
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -DARM_MATH_CM7 -D__FPU_PRESENT=1

# Include Paths
INCLUDES = -I. 
""")
            
            if self.config.include_cmsis_nn:
                f.write("INCLUDES += -ICMSIS/Include -ICMSIS-NN/Include\n")
            
            f.write(f"""
# Source Files
SRCS = {model_name}.c

# Object Files  
OBJS = $(SRCS:.c=.o)

# Libraries
LIBS = {' '.join(f'-l{lib}' for lib in self.target_specs.get('libs', []))}

# Linker Flags
LDFLAGS = -mcpu=$(MCU) -mthumb -mfpu=$(FPU) -mfloat-abi=hard
LDFLAGS += -specs=nano.specs -Wl,--gc-sections

# Build Rules
all: $(TARGET).elf $(TARGET).bin

$(TARGET).elf: $(OBJS)
\t$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
\t$(SIZE) $@

$(TARGET).bin: $(TARGET).elf
\t$(OBJCOPY) -O binary $< $@

%.o: %.c
\t$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
\trm -f $(OBJS) $(TARGET).elf $(TARGET).bin

flash: $(TARGET).bin
\t# Add your flash command here
\t@echo "Flash $(TARGET).bin to device"

.PHONY: all clean flash
""")
        
        return makefile_path
    
    def _generate_esp_idf_config(self, output_dir: str, model_name: str) -> str:
        """Generate ESP-IDF CMakeLists.txt."""
        cmake_path = os.path.join(output_dir, "CMakeLists.txt")
        
        with open(cmake_path, 'w') as f:
            f.write(f"""# CMakeLists.txt for {model_name} - ESP32-S3

cmake_minimum_required(VERSION 3.16)

set(COMPONENT_SRCS "{model_name}.c")
set(COMPONENT_ADD_INCLUDEDIRS ".")

register_component()

# ESP-NN optimizations
target_compile_definitions(${{COMPONENT_LIB}} PRIVATE CONFIG_ESP_NN_OPTIMIZED=1)
target_link_libraries(${{COMPONENT_LIB}} esp-nn esp-dsp)
""")
        
        return cmake_path
    
    def build_firmware(self, source_dir: str, output_file: str) -> str:
        """Compile the generated code."""
        print(f"Building firmware from {source_dir}...")
        
        if self.config.target == TargetDevice.ESP32_S3:
            return self._build_esp_idf(source_dir, output_file)
        else:
            return self._build_with_makefile(source_dir, output_file)
    
    def _build_with_makefile(self, source_dir: str, output_file: str) -> str:
        """Build using Makefile."""
        try:
            result = subprocess.run(
                ["make", "-C", source_dir],
                capture_output=True,
                text=True,
                check=True
            )
            print("Build successful!")
            print(result.stdout)
            
            # Copy output file
            built_binary = os.path.join(source_dir, "*.bin")
            if os.path.exists(built_binary):
                subprocess.run(["cp", built_binary, output_file], check=True)
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            print(f"stderr: {e.stderr}")
            raise
    
    def _build_esp_idf(self, source_dir: str, output_file: str) -> str:
        """Build using ESP-IDF."""
        try:
            # Run idf.py build
            result = subprocess.run(
                ["idf.py", "build"],
                cwd=source_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("ESP-IDF build successful!")
            
            # Copy output binary
            build_dir = os.path.join(source_dir, "build")
            binary_file = os.path.join(build_dir, "*.bin")
            if os.path.exists(binary_file):
                subprocess.run(["cp", binary_file, output_file], check=True)
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"ESP-IDF build failed: {e}")
            print(f"stderr: {e.stderr}")
            raise
    
    def flash(self, binary_path: str, port: str = "/dev/ttyUSB0") -> bool:
        """Flash firmware to target device."""
        print(f"Flashing {binary_path} to {self.config.target.value} on {port}...")
        
        try:
            if self.config.target == TargetDevice.ESP32_S3:
                cmd = ["esptool.py", "--port", port, "write_flash", "0x0", binary_path]
            else:
                cmd = ["st-flash", "write", binary_path, "0x8000000"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Flash successful!")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Flash failed: {e}")
            print(f"stderr: {e.stderr}")
            return False