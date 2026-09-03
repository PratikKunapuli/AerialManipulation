import torch
import os
import numpy as np

def export_model_to_c(model, export_dir, policy_rate, use_previous_action=False, ee_offset=None):
    """
    Exports a PyTorch model to a C header file. Assumes tanh activation for all but last layer. 
    
    Args:
        model (torch.nn.Module): The model to export.
        export_dir (str): The directory where the model will be saved.
    """
    print(f"Exporting model to {export_dir}")
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    header_path = f"{export_dir}/policy.h"
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract weights and biases
    weights = []
    biases = []
    layer_sizes = []
    
    # Process model parameters - only extract actor parameters
    with torch.no_grad():
        actor_params = {}
        print("Available model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
            # Only process actor parameters, skip critic and std
            if 'weight' in name or 'bias' in name:
                actor_params[name] = param.cpu().numpy()
                print(f"    -> Selected for export")
        
        print(f"\nSelected {len(actor_params)} actor parameters for export")
        
        # Sort parameters by layer number to ensure correct order
        # Extract layer numbers from parameter names like "actor.0.weight", "actor.2.bias"
        weight_layers = {}
        bias_layers = {}
        
        for name, param in actor_params.items():
            # Extract layer number from name (e.g., "actor.0.weight" -> 0)
            layer_num = int(name.split('.')[-2])
            
            if 'weight' in name:
                weight_layers[layer_num] = param
            elif 'bias' in name:
                bias_layers[layer_num] = param
        
        # Sort by layer number and add to lists
        for layer_num in sorted(weight_layers.keys()):
            weights.append(weight_layers[layer_num])
            if layer_num in bias_layers:
                biases.append(bias_layers[layer_num])
    
    print(f"Found {len(weights)} weight layers and {len(biases)} bias layers in actor network")
    
    # Determine layer sizes
    input_size = weights[0].shape[1]
    layer_sizes.append(input_size)
    
    for weight in weights:
        layer_sizes.append(weight.shape[0])
    
    num_layers = len(weights)
    
    # Generate C header file
    with open(header_path, 'w') as f:
        f.write("#ifndef POLICY_H\n")
        f.write("#define POLICY_H\n\n")
        f.write("#include <math.h>\n\n")
        
        # Write configuration constants
        f.write(f"#define NUM_LAYERS {num_layers}\n")
        f.write(f"#define INPUT_SIZE {input_size}\n")
        f.write(f"#define OUTPUT_SIZE {layer_sizes[-1]}\n\n")
        f.write(f"#define USE_PREVIOUS_ACTION {'1' if use_previous_action else '0'}\n\n")
        f.write(f"#define POLICY_RATE_HZ {policy_rate}\n\n")

        # Write layer rows and columns
        for i in range(num_layers):
            f.write(f"#define LAYER_{i}_ROWS {weights[i].shape[0]}\n")
            f.write(f"#define LAYER_{i}_COLS {weights[i].shape[1]}\n")

        f.write("#define EE_OFFSET_X " + (f"{ee_offset[0]:.3f}f\n" if ee_offset is not None else "0.0f\n"))
        f.write("#define EE_OFFSET_Y " + (f"{ee_offset[1]:.3f}f\n" if ee_offset is not None else "0.0f\n"))
        f.write("#define EE_OFFSET_Z " + (f"{ee_offset[2]:.3f}f\n" if ee_offset is not None else "0.0f\n\n"))

        # Write layer sizes array
        f.write("static const int layer_sizes[] = {")
        f.write(", ".join(map(str, layer_sizes)))
        f.write("};\n\n")
        
        # Write weights arrays
        for i, weight in enumerate(weights):
            f.write(f"static const float weights_{i}[] __attribute__((section(\".rodata\"))) = {{\n")
            # Flatten and write weights
            flat_weights = weight.flatten()
            for j, w in enumerate(flat_weights):
                if j % 8 == 0:
                    f.write("    ")
                f.write(f"{w:.8f}f")
                if j < len(flat_weights) - 1:
                    f.write(", ")
                if (j + 1) % 8 == 0 or j == len(flat_weights) - 1:
                    f.write("\n")
            f.write("};\n\n")
        
        # Write biases arrays
        for i, bias in enumerate(biases):
            f.write(f"static const float biases_{i}[] __attribute__((section(\".rodata\"))) = {{\n")
            f.write("    ")
            for j, b in enumerate(bias):
                f.write(f"{b:.8f}f")
                if j < len(bias) - 1:
                    f.write(", ")
            f.write("\n};\n\n")
        
        # Write the function definitions for usage:
        f.write("void policyLoadWeights(void);\n")
        f.write("void policyForward(const float* observations, float* actions);\n")
        f.write("void policyForwardRaw(const float* observations, float* actions);\n")
        f.write("#endif // POLICY_H\n")
    
    print(f"Model exported successfully to {header_path}")
    print(f"Network architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Total parameters: {sum(w.size for w in weights) + sum(b.size for b in biases)}")
    
    return header_path

def export_GC_gains(export_file_path, best_params, inertia_tensor=None):
    """ Export the gains of the DecoupledController to a file.
    
    Args:
        export_file_path (str): Path to save the exported gains.
        best_params (dict): Dictionary containing the best parameters for the controller.
        inertia_tensor (torch.Tensor, optional): Inertia tensor if needed for export.
    """

    return

def load_and_export_model(model_path, export_dir):
    """
    Convenience function to load a model from a file and export it to C.
    
    Args:
        model_path (str): Path to the saved PyTorch model (.pt or .pth file)
        export_dir (str): Directory where the C header will be saved
    """
    try:
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        
        # If the loaded object is a dict (state_dict), we need the actual model
        if isinstance(model, dict):
            print("Warning: Loaded a state dict. You need to provide the model architecture separately.")
            return None
        
        # Export to C
        return export_model_to_c(model, export_dir)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_test_c_file(header_path, export_dir):
    """
    Creates a simple test C file to verify the exported model works.
    
    Args:
        header_path (str): Path to the generated header file
        export_dir (str): Directory where test file will be saved
    """
    test_path = f"{export_dir}/test_policy.c"
    
    with open(test_path, 'w') as f:
        f.write('#include <stdio.h>\n')
        f.write('#include "policy.h"\n\n')
        f.write('int main() {\n')
        f.write(f'    float input[INPUT_SIZE] = {{0}}; // Initialize with zeros\n')
        f.write(f'    float output[OUTPUT_SIZE];\n\n')
        f.write('    // Set some test input values\n')
        f.write('    for (int i = 0; i < INPUT_SIZE; i++) {\n')
        f.write('        input[i] = 0.1f * i; // Simple test pattern\n')
        f.write('    }\n\n')
        f.write('    // Run inference\n')
        f.write('    neural_network_forward(input, output);\n\n')
        f.write('    // Print results\n')
        f.write('    printf("Input: ");\n')
        f.write('    for (int i = 0; i < INPUT_SIZE; i++) {\n')
        f.write('        printf("%.3f ", input[i]);\n')
        f.write('    }\n')
        f.write('    printf("\\n");\n\n')
        f.write('    printf("Output: ");\n')
        f.write('    for (int i = 0; i < OUTPUT_SIZE; i++) {\n')
        f.write('        printf("%.6f ", output[i]);\n')
        f.write('    }\n')
        f.write('    printf("\\n");\n\n')
        f.write('    return 0;\n')
        f.write('}\n')
    
    print(f"Test C file created: {test_path}")
    print("To compile and test:")
    print(f"gcc -o test_policy {test_path} -lm")
    print("./test_policy")
    
    return test_path

def validate_exported_model(original_model, header_path, num_tests=10):
    """
    Validates that the exported C model produces the same outputs as the original PyTorch model.
    
    Args:
        original_model (torch.nn.Module): The original PyTorch model
        header_path (str): Path to the exported header file
        num_tests (int): Number of random test cases to validate
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    import subprocess
    import tempfile
    import os
    
    # Get model dimensions
    original_model.eval()
    
    # Create a temporary C file for validation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        temp_c_file = f.name
        f.write(f'#include "{header_path}"\n')
        f.write('#include <stdio.h>\n')
        f.write('#include <stdlib.h>\n\n')
        f.write('int main(int argc, char* argv[]) {\n')
        f.write(f'    float input[INPUT_SIZE];\n')
        f.write(f'    float output[OUTPUT_SIZE];\n\n')
        f.write('    // Read input from command line\n')
        f.write('    for (int i = 0; i < INPUT_SIZE; i++) {\n')
        f.write('        input[i] = atof(argv[i + 1]);\n')
        f.write('    }\n\n')
        f.write('    neural_network_forward(input, output);\n\n')
        f.write('    // Print output\n')
        f.write('    for (int i = 0; i < OUTPUT_SIZE; i++) {\n')
        f.write('        printf("%.10f", output[i]);\n')
        f.write('        if (i < OUTPUT_SIZE - 1) printf(" ");\n')
        f.write('    }\n')
        f.write('    printf("\\n");\n')
        f.write('    return 0;\n')
        f.write('}\n')
    
    # Compile the test program
    temp_exe = temp_c_file.replace('.c', '')
    compile_cmd = f"gcc -o {temp_exe} {temp_c_file} -lm"
    
    try:
        subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        os.unlink(temp_c_file)
        return False
    
    print(f"Running validation with {num_tests} test cases...")
    
    validation_passed = True
    max_error = 0.0
    
    with torch.no_grad():
        for i in range(num_tests):
            # Get input size from the first actor layer
            actor_params = [p for name, p in original_model.named_parameters() if name.startswith('actor.') and 'weight' in name]
            if actor_params:
                input_size = actor_params[0].shape[1]
            else:
                print("Could not determine input size from actor parameters")
                return False
            
            test_input = torch.randn(1, input_size)
            
            # Get PyTorch output - need to extract only actor output
            if hasattr(original_model, 'actor'):
                # If model has an actor attribute
                pytorch_output = original_model.actor(test_input).squeeze().numpy()
            else:
                # Try to run forward pass and extract actor-like output
                with torch.no_grad():
                    full_output = original_model(test_input)
                    if isinstance(full_output, tuple):
                        # Assume first output is actor output
                        pytorch_output = full_output[0].squeeze().numpy()
                    else:
                        pytorch_output = full_output.squeeze().numpy()
            
            # Get C output
            input_args = ' '.join([str(x) for x in test_input.squeeze().numpy()])
            cmd = f"{temp_exe} {input_args}"
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                c_output = np.array([float(x) for x in result.stdout.strip().split()])
                
                # Compare outputs
                error = np.max(np.abs(pytorch_output - c_output))
                max_error = max(max_error, error);
                
                if error > 1e-5:  # Tolerance for floating point comparison
                    print(f"Test {i+1} FAILED: max error = {error}")
                    print(f"PyTorch: {pytorch_output}")
                    print(f"C:       {c_output}")
                    validation_passed = False
                else:
                    print(f"Test {i+1} PASSED: max error = {error}")
                    
            except subprocess.CalledProcessError as e:
                print(f"C program execution failed: {e}")
                validation_passed = False
                break
    
    # Cleanup
    os.unlink(temp_c_file)
    os.unlink(temp_exe)
    
    return validation_passed

def extract_actor_from_model(model):
    """
    Helper function to extract just the actor network from an RL model.
    
    Args:
        model: The full RL model (with actor and critic)
    
    Returns:
        A new model containing only the actor weights
    """
    import torch.nn as nn
    
    # Extract actor parameters
    actor_state_dict = {}
    layer_sizes = []
    
    # Get all actor parameters
    for name, param in model.named_parameters():
        if name.startswith('actor.') and ('weight' in name or 'bias' in name):
            # Remove 'actor.' prefix for cleaner state dict
            clean_name = name.replace('actor.', '')
            actor_state_dict[clean_name] = param.data
    
    # Determine layer architecture from weights
    weight_layers = {}
    for name, param in actor_state_dict.items():
        if 'weight' in name:
            layer_num = int(name.split('.')[0])
            weight_layers[layer_num] = param.shape
    
    # Build layer sizes
    sorted_layers = sorted(weight_layers.keys())
    if sorted_layers:
        input_size = weight_layers[sorted_layers[0]][1]  # Input features of first layer
        layer_sizes.append(input_size)
        
        for layer_num in sorted_layers:
            layer_sizes.append(weight_layers[layer_num][0])  # Output features
    
    # Create a new sequential model with the actor architecture
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        # Add tanh activation for all but the last layer
        if i < len(layer_sizes) - 2:
            layers.append(nn.Tanh())
    
    actor_model = nn.Sequential(*layers)
    
    # Load the weights
    actor_model.load_state_dict(actor_state_dict)
    
    print(f"Extracted actor model with architecture: {' -> '.join(map(str, layer_sizes))}")
    
    return actor_model

if __name__ == "__main__":
    export_model_dir = "../rl/logs/rsl_rl/sysID/2025-07-28_18-05-11_manipulator_CTBR_sysid_corrected_model_100Hz/exported/"
    
    # Example usage:
    # Option 1: If you have a model file (.pt or .pth)
    model_path = "../rl/logs/rsl_rl/sysID/2025-07-28_18-05-11_manipulator_CTBR_sysid_corrected_model_100Hz/model.pt"
    
    # Try to load and export the model
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        header_path = load_and_export_model(model_path, export_model_dir)
        if header_path:
            # Create a test file
            create_test_c_file(header_path, export_model_dir)
    else:
        print(f"Model file not found: {model_path}")
        print("Please provide the correct path to your trained model.")
        print("\nExample usage:")
        print("1. If you have a model instance:")
        print("   export_model_to_c(your_model, export_dir)")
        print("2. If you have a saved model file:")
        print("   load_and_export_model('path/to/model.pt', export_dir)")
        
        # Example with a dummy model for demonstration
        print("\nCreating a dummy RL model for demonstration...")
        import torch.nn as nn
        
        # Create a dummy RL model with both actor and critic
        class DummyRLModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Actor network
                self.actor = nn.Sequential(
                    nn.Linear(12, 64),   # Input layer (e.g., 12 state variables)
                    nn.Tanh(),
                    nn.Linear(64, 32),   # Hidden layer
                    nn.Tanh(),
                    nn.Linear(32, 4)     # Output layer (e.g., 4 motor commands)
                )
                # Critic network (will be ignored during export)
                self.critic = nn.Sequential(
                    nn.Linear(12, 64),
                    nn.Tanh(),
                    nn.Linear(64, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)     # Value output
                )
                # Standard deviation parameter (will be ignored)
                self.std = nn.Parameter(torch.ones(4) * 0.1)
            
            def forward(self, x):
                return self.actor(x), self.critic(x)
        
        # Create the model and add parameter names similar to RSL-RL format
        dummy_rl_model = DummyRLModel()
        
        # Rename parameters to match RSL-RL naming convention
        state_dict = dummy_rl_model.state_dict()
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('actor.'):
                new_state_dict[name] = param
            elif name.startswith('critic.'):
                new_state_dict[name] = param
            elif name == 'std':
                new_state_dict[name] = param
        
        dummy_rl_model.load_state_dict(new_state_dict)
        
        print("Dummy RL model created with actor, critic, and std parameters")
        
        # Export the dummy model (only actor will be exported)
        header_path = export_model_to_c(dummy_rl_model, export_model_dir)
        if header_path:
            create_test_c_file(header_path, export_model_dir)
            
            # Optional: Extract just the actor for validation
            actor_only = extract_actor_from_model(dummy_rl_model)
            print("\nValidating exported model against extracted actor...")
            validate_exported_model(actor_only, header_path, num_tests=3)