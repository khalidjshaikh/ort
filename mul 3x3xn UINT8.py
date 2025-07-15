# python onnx onnxruntime matmul 3x3x1000000 matrix ir_version 10 opset_version 22

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
import onnxruntime as ort
import numpy as np

start_time = time.perf_counter()

# 1. Define the ONNX Model
# Assuming two 3x3 matrices are multiplied across 1,000,000 batches
batch_size = int(1e8)
input_shape = [batch_size, 3, 3]
# input_shape = [3, 3, batch_size]

# Define input tensors
X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.UINT8, input_shape)
Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.UINT8, input_shape)

# Define output tensor
Z = onnx.helper.make_tensor_value_info('Z', onnx.TensorProto.UINT8, input_shape)

# Create MatMul node
matmul_node = onnx.helper.make_node(
    'Mul',
    # 'Sub',
    # 'Add',
    inputs=['X', 'Y'],
    outputs=['Z'],
)

# Create graph
graph_def = onnx.helper.make_graph(
    [matmul_node],
    'matmul_graph',
    [X, Y],
    # []
    [Z],
)

# Create model
model_def = onnx.helper.make_model(graph_def, producer_name='matmul_example')
model_def.ir_version = 10
model_def.opset_import[0].version = 22

# 2. Save the ONNX Model
onnx.save(model_def, 'matmul_model.onnx')

# 3. Load and Run with ONNX Runtime
session = ort.InferenceSession('matmul_model.onnx',
                            providers=["QNNExecutionProvider"],
                            provider_options=[
                                {
                                    # "backend_path" : "QnnCpu.dll",
                                    # "backend_path" : "QnnGpu.dll",
                                    # "backend_path" : "QnnHtp.dll",
                                    # "backend_type" : "cpu"
                                    # "backend_type" : "gpu"
                                    # "backend_type" : "htp"
                                    # "backend_type" : "saver"
                                    # "profiling_level": "detailed",
                                    # "profiling_file_path": "x.csv"
                                 }
                            ] # Provide path to Htp dll in QNN SDK
)

# Prepare input data (example random data)
input_x = np.random.randint(0, 16, size=input_shape, dtype=np.uint8)
input_y = np.random.randint(0, 16, size=input_shape, dtype=np.uint8)

# Run inference
outputs = session.run(
    None,
    {'X': input_x, 'Y': input_y}
)

result_z = outputs[0]

print("Result shape:", result_z.shape)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(input_x)
print(input_y)
print(outputs)

print(input_shape, ": {:.2e}".format(prod(input_shape)))
print("{:.2e}".format(prod(input_shape)/elapsed_time))
print(elapsed_time)
