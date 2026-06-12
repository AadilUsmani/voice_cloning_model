import torch

# 1. Load the model into your CPU memory
checkpoint = torch.load("step_40000_test_0.pt", map_location="cpu")

# 2. Find the model's weights (the state_dict)
# Sometimes it's nested under a 'model' or 'state_dict' key
weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

# 3. Print the names and shapes of the first 5 layers
print("--- ARCHITECTURE LOOKUP ---")
for i, key in enumerate(list(weights.keys())[:5]):
    print(f"Layer: {key} | Shape: {weights[key].shape}")

# 4. Print the ACTUAL weights (numbers) of the very first layer
first_layer = list(weights.keys())[0]
print(f"\n--- RAW WEIGHTS FOR: {first_layer} ---")
print(weights[first_layer])