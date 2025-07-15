import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper, TensorProto
import time # Import time for performance measurement

import sys
import onnxruntime
import numpy
print(f"Python version: {sys.version}")
print(f"ONNX version: {onnx.__version__}")
print(f"ONNX Runtime version: {onnxruntime.__version__}")
print(f"NumPy version: {numpy.__version__}")

# ... (Previous code for creating and saving the ONNX model remains the same) ...
# Define the graph
input_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [3, 3])
input_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [3, 3])
output_c = helper.make_tensor_value_info('output_c', TensorProto.FLOAT, [3, 3])

add_node = helper.make_node("Add", ["input_a", "input_b"], ["output_c"])

graph = helper.make_graph([add_node], "matrix_addition", [input_a, input_b], [output_c])

# Create and save the ONNX model
opset_imports = [helper.make_operatorsetid("", 22)]
model = helper.make_model(graph, producer_name="matrix_addition_model", opset_imports=opset_imports)
model.ir_version = 10

onnx_model_path = "matrix_addition.onnx"
onnx.save(model, onnx_model_path)
print(f"ONNX model saved to: {onnx_model_path}")

# --- Performance considerations for a large loop ---

# Create an ONNX Runtime inference session
# You might want to explore session options for optimization, but start simple.
sess_options = rt.SessionOptions()
# sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL # Example option

sess = rt.InferenceSession(onnx_model_path, 
                        providers=["QNNExecutionProvider"],
                        provider_options=[{"backend_path": "QnnHtp.dll"}] # Provide path to Htp dll in QNN SDK
                        #    providers=rt.get_available_providers(), 
                        #    sess_options=sess_options
                        )

# Prepare input data outside the loop
matrix_a = np.random.randint(10, size=(3, 3)).astype(np.float32)
matrix_b = np.random.randint(10, size=(3, 3)).astype(np.float32)

input_feed = {
    "input_a": matrix_a,
    "input_b": matrix_b
}

# Loop for inference
num_iterations = 1_000_000_000 # 1 billion iterations
num_iterations = int(1e5)

print(f"\nStarting {num_iterations} inference runs...")
start_time = time.time()

# 45 TOPS
# 2 GHz (BOPS)
# 1e5 / 56.623
# 1766.066792646098
for _ in range(num_iterations):
    # Run inference without recreating the session or input feed
    # output = sess.run(None, input_feed) # Basic run
    # If performance is critical, consider IO binding to avoid data copies
    io_binding = sess.io_binding()
    io_binding.bind_cpu_input('input_a', matrix_a)
    io_binding.bind_cpu_input('input_b', matrix_b)
    io_binding.bind_output('output_c')
    sess.run_with_iobinding(io_binding)
    # result_matrix = io_binding.copy_outputs_to_cpu()[0] # Get output if needed
    # For performance testing, you might skip copying the output to CPU in the loop
    if (_ % int(1e4) == 0): print(_)


end_time = time.time()
print(f"Finished {num_iterations} inference runs.")

duration = end_time - start_time
print(f"Total duration: {duration:.2f} seconds")
print(f"Average time per inference: {duration / num_iterations:.6f} seconds")

# Optional: Verify the result after the loop (e.g., after a few runs or the last run)
# print("\nResult Matrix (ONNX Runtime):\n", result_matrix)
# expected_result = matrix_a + matrix_b
# print("\nExpected Result (NumPy):\n", expected_result)
# assert np.array_equal(result_matrix, expected_result)
# print("\nONNX Runtime result matches NumPy result.")

