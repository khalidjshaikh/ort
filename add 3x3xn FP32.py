# python onnx onnxruntime add 3x3x1000000 matrix ir_version 10 opset_version 22
# pip install onnx onnxruntime numpy

import onnx
import onnxruntime
import numpy
from numpy import *
import sys
import time

print(f"Python version: {sys.version}")
print(f"ONNX version: {onnx.__version__}")
print(f"ONNX Runtime version: {onnxruntime.__version__}")
print(f"NumPy version: {numpy.__version__}")

import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

start_time = time.perf_counter()

# n = int(1e8)
n = int(1e7)

# n = int(1e7)
# n = int(1e6)
# Define input shapes and types
input_shape = [3, 3, n]
print(input_shape, ": {:.2e}".format(prod(input_shape)))

input_type = TensorProto.FLOAT

# Create ValueInfoProto for inputs and output
X = helper.make_tensor_value_info("X", input_type, input_shape)
Y = helper.make_tensor_value_info("Y", input_type, input_shape)
Z = helper.make_tensor_value_info("Z", input_type, input_shape)

# Create the 'Add' node
node_def = helper.make_node(
    "Add",
    ["X", "Y"],
    ["Z"],
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],
    "add-model",
    [X, Y],
    [Z],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name="onnx-example")

# Set the desired IR version and opset version
model_def.ir_version = 10  # IR version 10
model_def.opset_import[0].version = 22  # Opset version 22

# Check the model
onnx.checker.check_model(model_def)
print("The ONNX model is checked!")

# Save the model
onnx.save(model_def, "add_large_matrices.onnx")
print("Model saved to add_large_matrices.onnx")

import onnxruntime as ort
import numpy as np

# Load the ONNX model
sess = ort.InferenceSession("add_large_matrices.onnx",
                            # providers=["QNNExecutionProvider"]
                            )

# Create dummy input data
input_data_X = np.random.rand(3, 3, n).astype(np.float32)
input_data_Y = np.random.rand(3, 3, n).astype(np.float32)

# Run inference
outputs = sess.run(None, {"X": input_data_X, "Y": input_data_Y})

# Get the output (result of addition)
output_Z = outputs[0]

# Verify the output
expected_output = input_data_X + input_data_Y
print("Output matches expected:", np.allclose(output_Z, expected_output))

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("{:.2e}".format(prod(input_shape)/elapsed_time))
print(elapsed_time)
