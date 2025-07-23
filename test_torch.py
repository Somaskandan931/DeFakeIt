import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.current_device()) # Should print device index like 0
print(torch.cuda.get_device_name(0))  # Should print your GPU name "NVIDIA GeForce GTX 1650"
