from torch import nn
import torch

class MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return self.coef * loss

def rgb_to_gray(image):
    gray_image = torch.sum(image * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), dim=1, keepdim=True)
    return gray_image

def compute_gradient_loss(pred, gt, mask):
    pred = rgb_to_gray(pred)
    gt = rgb_to_gray(gt)

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

    def compute_gradient(conv_kernel, tensor):
        return torch.nn.functional.conv2d(tensor.repeat(1, 3, 1, 1), conv_kernel.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), padding=1) / 3

    gradient_a_x = compute_gradient(sobel_kernel_x, pred)
    gradient_a_y = compute_gradient(sobel_kernel_y, pred)
    gradient_b_x = compute_gradient(sobel_kernel_x, gt)
    gradient_b_y = compute_gradient(sobel_kernel_y, gt)

    pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
    gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

    gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1, keepdim=True)[mask].sum() / (mask.sum() + 1e-8)

    return gradient_difference

loss_dict = {'mse': MSELoss}
