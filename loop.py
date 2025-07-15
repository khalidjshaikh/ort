import onnxruntime as ort
ort.set_default_logger_severity(3)

import sys
import onnx
import onnxruntime as ort

print(f"Python version: {sys.version}")  # Prints the full Python version string.
print(f"ONNX version: {onnx.__version__}") # Prints the ONNX version number.
print(f"ONNX Runtime version: {ort.__version__}") # Prints the ONNX Runtime version number.

import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Model parameters
ir_version = 10
opset_version = 22
loop_count = 1_000_000_000

# Inputs: trip count (1B), condition (True), and initial loop var (0)
trip_count = helper.make_tensor_value_info('trip_count', onnx.TensorProto.INT64, [])
cond_in = helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
loop_var_in = helper.make_tensor_value_info('loop_var_in', onnx.TensorProto.INT64, [])

# Outputs
loop_var_out = helper.make_tensor_value_info('loop_var_out', onnx.TensorProto.INT64, [])

# Loop body graph
iter_count = helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
cond = helper.make_tensor_value_info('cond', onnx.TensorProto.BOOL, [])
loop_var = helper.make_tensor_value_info('loop_var', onnx.TensorProto.INT64, [])
next_cond = helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
next_loop_var = helper.make_tensor_value_info('loop_var_out', onnx.TensorProto.INT64, [])

# Body logic: loop_var += 1
add_node = helper.make_node('Add', ['loop_var', 'const_one'], ['loop_var_out'])
identity_node = helper.make_node('Identity', ['cond'], ['cond_out'])

const_one = helper.make_tensor('const_one', onnx.TensorProto.INT64, [], [1])

loop_body = helper.make_graph(
    [add_node, identity_node],
    'loop_body',
    [iter_count, cond, loop_var],
    [next_cond, next_loop_var],
    initializer=[const_one]
)

# Main loop node
loop_node = helper.make_node(
    'Loop',
    ['trip_count', 'cond_in', 'loop_var_in'],
    ['loop_var_out'],
    body=loop_body
)

graph = helper.make_graph(
    [loop_node],
    'loop_graph',
    [trip_count, cond_in, loop_var_in],
    [loop_var_out]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
model.ir_version = ir_version

onnx.save(model, 'loop_1b.onnx')
print("Saved loop model to loop_1b.onnx")


import onnxruntime as ort
import numpy as np
from numpy import *

sess = ort.InferenceSession("loop_1b.onnx")

inputs = {
    # "trip_count": np.int64(1_000_000_000),
    # "trip_count": np.int64(int(1e12)),
    "trip_count": array([int64(1e7)]),
    # "cond_in": "true", # np.bool_(True),
    "cond_in": array([np.bool_(True)]),
    # "loop_var_in": np.int64(0),
    "loop_var_in": array([int64(0)])
}

print(inputs)

print("Running inference...")
output = sess.run(None, inputs)
print("Loop result:", output[0])

