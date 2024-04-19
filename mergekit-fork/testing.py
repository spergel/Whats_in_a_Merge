!git clone https://huggingface.co/JSpergel/test_tiny_mixtral_only_router
from transformers import AutoTokenizer
import torch

moe_router_weights = torch.load("/content/test_tiny_mixtral_only_router/moe_router_weights.pth")
tokenizer = AutoTokenizer.from_pretrained("JSpergel/test_tiny_mixtral_only_router")

test = tokenizer("Does this tokenize?")
tensor = torch.tensor(test['input_ids'])

# Ensure that the data type is the same for both the tensor and the weights
tensor = tensor.float()
weights = moe_router_weights['model.layers.0.block_sparse_moe.gate.weight'][0,].float()

print(weights)
# Now perform the matrix multiplication
output = torch.matmul(tensor, weights)