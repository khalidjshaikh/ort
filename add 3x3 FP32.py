import onnx
import onnxruntime as ort
import numpy as np

# 1. Create the ONNX model
model_input_name1 = "input_matrix_a"
model_input_name2 = "input_matrix_b"
model_output_name = "output_matrix_c"

# Define the input and output tensors
input_a = onnx.helper.make_tensor_value_info(model_input_name1, onnx.TensorProto.FLOAT, [3, 3])
input_b = onnx.helper.make_tensor_value_info(model_input_name2, onnx.TensorProto.FLOAT, [3, 3])
output_c = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [3, 3])

# Create an Add node
node = onnx.helper.make_node(
    "Add",
    inputs=[model_input_name1, model_input_name2],
    outputs=[model_output_name]
)

# Create the graph (inputs, outputs, and nodes)
graph = onnx.helper.make_graph(
    [node],
    "add_graph",
    [input_a, input_b],
    [output_c]
)

# Create the ONNX model
onnx_model = onnx.helper.make_model(
    graph, 
    producer_name="onnx-example",
    opset_imports=[onnx.helper.make_opsetid("", 22)]
)
onnx_model.ir_version = 10

# Save the model to a file
onnx.save(onnx_model, "add_model.onnx")

# 2. Load the ONNX model
# session = ort.InferenceSession("add_model.onnx", providers=ort.get_available_providers())
# providers = ["CPUExecutionProvider"]
providers = ["QNNExecutionProvider"]
# session = ort.InferenceSession("add_model.onnx", providers=providers)
session = ort.InferenceSession("add_model.onnx")
# session = ort.InferenceSession(onnx_model)

# 3. Prepare NumPy arrays
matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
matrix_b = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32)

# 4. Run inference
inputs = {model_input_name1: matrix_a, model_input_name2: matrix_b}
outputs = session.run([model_output_name], inputs)

# Print the result
result_matrix = outputs[0]
print(matrix_a)
print(matrix_b)
print("Result of matrix addition:\n", result_matrix)