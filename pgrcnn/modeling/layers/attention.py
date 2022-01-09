import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
from detectron2.layers import Conv2d, get_norm

class AttentionConv(nn.Module):
    """
    @misc{ramachandran2019standalone,
      title={Stand-Alone Self-Attention in Vision Models},
      author={Prajit Ramachandran and Niki Parmar and Ashish Vaswani and Irwan Bello and Anselm Levskaya and Jonathon Shlens},
      year={2019},
      eprint={1906.05909},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, norm='SyncBN'):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
        # increase the kernel size for larger reception field of keypoints
        self.key_conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, norm=get_norm(norm, out_channels))
        self.query_conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, norm=get_norm(norm, out_channels))
        self.value_conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, norm=get_norm(norm, out_channels))

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)



if __name__ == "__main__":
    temp = torch.randn((2, 3, 32, 32))
    conv = AttentionConv(3, 16, kernel_size=3, padding=1)
    print(conv(temp).size())