import torch
import einops
import copy

# Synthetic Data Creation
batch = {
    "feature_1": torch.randn(2, 3, 4),  # (batch_size=2, seq_len=3, feature_dim=4)
    "feature_2": torch.randn(2, 3, 5),  # (batch_size=2, seq_len=3, feature_dim=5)
}
config = {
    "input_shapes": ["feature_1", "feature_2"]  # Keys to process
}

# Copy the batch for verification
original_batch = copy.deepcopy(batch)

# Flatten the sequence dimension into the batch dimension
batch_size = batch["feature_1"].shape[0]  # Assume all features share the same batch size
for feature in config["input_shapes"]:
    batch[feature] = einops.rearrange(batch[feature], "b s ... -> (b s) ...")

# Simulate the encoder output
# Here, we mock the output as a concatenation of feature dimensions for simplicity
# Replace with your actual encoder logic
global_cond_all = torch.cat([batch[feature] for feature in config["input_shapes"]], dim=-1)

# Reshape the output back to batch and sequence dimensions
reconstructed_output = einops.rearrange(global_cond_all, "(b s) d -> b (s d)", b=batch_size)

print(original_batch)
print(reconstructed_output)

# Print Shapes for Verification
print("Original batch shapes:")
for key, value in original_batch.items():
    print(f"{key}: {value.shape}")

print("\nBatch after flattening sequence dimension:")
for key, value in batch.items():
    print(f"{key}: {value.shape}")

print(f"\nEncoder output shape: {global_cond_all.shape}")
print(f"\nReconstructed output shape: {reconstructed_output.shape}")

# Validate Correspondences
# Check that the reconstructed output matches the original shapes
assert reconstructed_output.shape[0] == original_batch["feature_1"].shape[0], "Batch size mismatch!"
assert reconstructed_output.shape[1] == original_batch["feature_1"].shape[1], "Sequence length mismatch!"
assert reconstructed_output.shape[2] == sum([v.shape[-1] for v in original_batch.values()]), "Feature dimension mismatch!"

print("\nVerification passed: All shapes align correctly!")
