import torch

# Check if CUDA is available and print device name
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Initialize tensors directly on the GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.tensor([1, 2, 3], device=device)  # Create tensor on GPU
B = torch.tensor([4, 5, 6], device=device)  # Create tensor on GPU

# Perform element-wise multiplication on GPU
C = A * B

print(C)
