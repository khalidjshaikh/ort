import onnxruntime as ort

# Create an InferenceSession (you can use a dummy model or a real one)
# For demonstration, we'll create a dummy session
# In a real scenario, you would load your ONNX model:
# session = ort.InferenceSession("your_model.onnx")

# To get available providers without loading a model, you can use get_available_providers()
available_providers = ort.get_available_providers()
print("Available ONNX Runtime Execution Providers:")
for provider in available_providers:
    print(f"- {provider}")