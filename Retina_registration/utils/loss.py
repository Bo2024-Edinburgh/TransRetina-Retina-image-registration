import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()


"""
General implementation of local normalized cross correlation loss
adapted for 1D/2D/3D data. 
"""
class ncc_loss_fun:
    def __init__(self, device="cpu", win=None):
        self.win = win
        print("[========= win info]", win)
        self.device = device

    def ncc_loss(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "Expected input dimension 1 to 3, got: %d" % ndims

        # Set window size
        if self.win is None:
            win = [9] * ndims
        else:
            win = [self.win] * ndims
        
        sum_filt = torch.ones([1, 1, *win]).to(self.device)
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride, padding = (1,), (pad_no,)
        elif ndims == 2:
            stride, padding = (1,1), (pad_no, pad_no)
        else:
            stride, padding = (1,1,1), (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, f'conv{ndims}d')

        I2, J2, IJ = Ii * Ii, Ji * Ji, Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I, u_J = I_sum / win_size, J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)



"""
Implements standard MSE-based losses including L2 and L1 versions.
Typically used for auxiliary supervision in registration tasks.
"""
class MSE:
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2) * 20

    def loss_sum(self, y_true, y_pred):
        return torch.sum((y_true - y_pred) ** 2)

    def loss_l1(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred)) * 5


"""
Computes a simple 2D smoothness loss by taking differences in
the x and y directions and applying an L2 penalty.
"""
class smooth_loss:
    def smooothing_loss_2d(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dy, dx = dy * dy, dx * dx
        return (torch.mean(dx) + torch.mean(dy)) / 2.0


"""
VoxelMorph-style general N-D gradient loss with optional L1 or L2 norm.
Supports 1D, 2D, or 3D deformation fields.

Args:
    pred: Tensor of shape [B, C, ...] representing the deformation field.
"""
class Grad(nn.Module):
    def __init__(self, penalty='l2'):
        super(Grad, self).__init__()
        self.penalty = penalty

    def _diffs(self, y):
        ndims = y.ndimension() - 2
        df = [None] * ndims

        for i in range(ndims):
            d = i + 2
            print("smooth debug2:", i, d, y.shape)
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
            print("smooth debug3:", i, d, y.shape)
            dfi = y[1:, ...] - y[:-1, ...]
            print("smooth debug4:", dfi.shape)
            df[i] = dfi.permute(*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2))
            print("smooth debug5:", dfi[i].shape)
        
        return df

    def forward(self, pred):
        ndims = pred.ndimension() - 2
        df = Variable(torch.zeros(1).cuda() if pred.is_cuda else torch.zeros(1))
        print("smooth debug1:", df)

        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', f'Invalid penalty: {self.penalty}'
                df += f.pow(2).mean() / ndims
        return df


"""
Computes the Dice similarity coefficient for segmentation evaluation.
This is not a loss but a metric, often used to evaluate registration or segmentation overlap.

Args:
    y_true: Ground truth tensor.
    y_pred: Predicted tensor.

Returns:
    The mean Dice score across the batch.
"""
def dice_score_2d(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return dice
