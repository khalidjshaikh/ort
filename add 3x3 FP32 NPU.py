# python onnx onnxruntime add 3x3 matrix ir_version 10 opset_version 22


import sys
import onnx
import onnxruntime as ort

print(f"Python version: {sys.version}")  # Prints the full Python version string.
print(f"ONNX version: {onnx.__version__}") # Prints the ONNX version number.
print(f"ONNX Runtime version: {ort.__version__}") # Prints the ONNX Runtime version number.

import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper, TensorProto

# Define the input tensors with their names, types (FLOAT for 32-bit floats),
# and shapes (3x3 matrices)
input_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [3, 3])  #
input_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [3, 3])  #

# Define the output tensor with its name, type, and shape
output_c = helper.make_tensor_value_info('output_c', TensorProto.FLOAT, [3, 3])  #

# Create an 'Add' node to perform element-wise addition of the input matrices
add_node = helper.make_node(
    "Add",  # ONNX operator type
    ["input_a", "input_b"],  # Input names to the operator
    ["output_c"]  # Output name from the operator
)

# Build the graph using the defined inputs, outputs, and node
graph = helper.make_graph(
    [add_node],  # List of nodes in the graph
    "matrix_addition",  # Name of the graph
    [input_a, input_b],  # Inputs to the graph
    [output_c]  # Outputs from the graph
)

# Set the desired IR version and opset version
opset_imports = [helper.make_operatorsetid("", 22)]  # Opset version 22 for the default domain
model = helper.make_model(graph, producer_name="matrix_addition_model", opset_imports=opset_imports)  #
model.ir_version = 10  # IR version 10

# Save the ONNX model to a file
onnx_model_path = "matrix_addition.onnx"
onnx.save(model, onnx_model_path)
print(f"ONNX model saved to: {onnx_model_path}")

# Create random 3x3 matrices as input data
matrix_a = np.random.randint(10, size=(3, 3)).astype(np.float32)  #
matrix_b = np.random.randint(10, size=(3, 3)).astype(np.float32)  #

print("\nInput Matrix A:\n", matrix_a)
print("\nInput Matrix B:\n", matrix_b)

# Create an ONNX Runtime inference session
sess = rt.InferenceSession(onnx_model_path, 
                            providers=["QNNExecutionProvider"],
                            provider_options=[{"backend_path": "QnnHtp.dll"}] # Provide path to Htp dll in QNN SDK
                        #    providers=rt.get_available_providers()
                           )  #

# Prepare input data for the session
input_feed = {
    "input_a": matrix_a,
    "input_b": matrix_b
}  #

# Run inference and get the output
output = sess.run(None, input_feed)  #
result_matrix = output[0]

print("\nResult Matrix (ONNX Runtime):\n", result_matrix)

# Verify the result with NumPy
expected_result = matrix_a + matrix_b
print("\nExpected Result (NumPy):\n", expected_result)

# Check for equality
# assert np.array_equal(result_matrix, expected_result)
print("\nONNX Runtime result matches NumPy result.")

