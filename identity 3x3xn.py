import time
start_time = time.perf_counter()

from numpy import * 

import onnx
import onnxruntime as rt
import numpy as np
from onnx import helper, TensorProto

# 1. Define the input tensor
n=int(1e8)
input_shape = [3, 3, n]
input_name = 'input_tensor'
output_name = 'output_tensor'

# Define the input and output tensors for the graph
input_tensor = helper.make_tensor_value_info(input_name, TensorProto.UINT8, input_shape)
output_tensor = helper.make_tensor_value_info(output_name, TensorProto.UINT8, input_shape)

# 2. Create the Identity node
identity_node = helper.make_node(
    'Identity',
    inputs=[input_name],
    outputs=[output_name]
)

# 3. Construct the ONNX graph
graph = helper.make_graph(
    [identity_node],
    'identity_graph',
    [input_tensor],
    [output_tensor]
)

# 4. Create the ONNX model
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 22)])

# 5. Saving the model
onnx_model_path = "identity_model.onnx"
onnx.save(model, onnx_model_path)
print(f"ONNX model saved to {onnx_model_path}")

# 6. Running inference with ONNX Runtime
# Create a dummy input data
dummy_input_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

# Load the ONNX model with ONNX Runtime
sess = rt.InferenceSession(onnx_model_path,
                            providers=["QNNExecutionProvider"],
                            provider_options=[{
                                "backend_path" : "QnnHtp.dll",
                            }]
)

# Run inference
output_data = sess.run([output_name], {input_name: dummy_input_data})[0]

# Verify the output
assert np.array_equal(dummy_input_data, output_data)
print("ONNX Runtime inference successful and output matches input.")

print(dummy_input_data)
print(output_data)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(input_shape, ": {:.2e}".format(prod(input_shape)))
print("{:.2e}".format(prod(input_shape)/elapsed_time))
print(elapsed_time)
