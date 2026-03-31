import torch

# 1. Load the saved weights (mapping to CPU so you don't need a local GPU)
weights = torch.load("encoder_best.pth", map_location=torch.device('cpu'), weights_only=True)

# 2. Print all the layers and their sizes
print("🧠 MODEL ARCHITECTURE & TENSORS:")
for layer_name, tensor in weights.items():
    print(f"Layer: {layer_name:<30} | Shape: {tensor.shape}")

# 3. Peek at the actual learned numbers in the very first LSTM layer
first_layer = list(weights.keys())[0]
print(f"\n🔍 PEEKING INSIDE: {first_layer}")
print(weights[first_layer][:5]) # Prints the first 5 numbers of that layer