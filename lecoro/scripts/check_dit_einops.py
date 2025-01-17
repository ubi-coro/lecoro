import torch
import einops
import copy

# Define synthetic batch similar to real input
batch = {
    "feature_1": torch.randn(3, 4, 5),  # (batch_size=3, seq_len=4, feature_dim=4)
    "feature_2": torch.randn(3, 4, 5),  # (batch_size=3, seq_len=4, feature_dim=5)
}

# Mock configuration
class MockConfig:
    input_shapes = batch.keys()
    token_dim = 6  # Simulating the encoder's output token dim

config = MockConfig()

# Mock Encoder: Simulates the behavior of `obs_encoder`
class MockEncoder:
    def __call__(self, batch):
        """Simulate encoding by concatenating feature tensors along the last axis."""
        features = [batch[f] for f in config.input_shapes]
        encoded = torch.stack(features, dim=1)  # Concatenate feature dims
        return encoded  # Output shape should be `(b*t, s, feature_dim)`

# Instantiate the mock encoder
obs_encoder = MockEncoder()

# Function to verify _prepare_global_conditioning
def _prepare_global_conditioning(batch):
    """Encode image features and concatenate them all together along with the state vector."""
    batch = copy.copy(batch)
    batch_size = batch["feature_1"].shape[0]

    # Absorb obs_history dim into batch dimension
    for feature in config.input_shapes:
        batch[feature] = einops.rearrange(batch[feature], "b t ... -> (b t) ...")

    global_cond_all = obs_encoder(batch)  # Simulated (b*t, s, feature_dim)

    # Separate obs_history dim back out and absorb into token dim, effectively concatenating them
    return einops.rearrange(global_cond_all, "(b t) s d -> b (t s) d", b=batch_size)

# Compute `_prepare_global_conditioning` on the entire batch
global_cond_batch = _prepare_global_conditioning(batch)

# Compute `_prepare_global_conditioning` by processing each sample individually
batch_size = batch["feature_1"].shape[0]
global_cond_list = []
for i in range(batch_size):
    new_batch = {key: batch[key][i, None, ...] for key in batch}  # Extract individual sample
    global_cond_list.append(_prepare_global_conditioning(new_batch))

# Stack individual results
global_cond_individual = torch.cat(global_cond_list, dim=0)

# Validate that the two approaches yield the same result
assert torch.allclose(global_cond_batch, global_cond_individual), "Mismatch detected!"

# Print Results
print("\nGlobal Conditioning (Batch Processing):\n", global_cond_batch)
print("\nGlobal Conditioning (Individual Processing & Stacking):\n", global_cond_individual)
print("\nTest Passed: Global conditioning aligns correctly!")
