import torch
import torch.nn.functional as F

src = torch.arange(25, dtype=torch.float).reshape(1, 1, 5, 5).requires_grad_()  # 1 x 1 x 5 x 5 with 0 ... 25
indices = torch.tensor([[-1, -1], [0, 0]], dtype=torch.float).reshape(1, 1, -1, 2)  # 1 x 1 x 2 x 2
output = F.grid_sample(src, indices)
print(output)  # tensor([[[[  0.,  12.]]]])