import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from detectron2.layers import Conv2d, get_norm

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

    # def forward(self, x):
    #     feat1 = self.conv5a(x)
    #     sa_feat = self.sa(feat1)
    #     sa_conv = self.conv51(sa_feat)
    #     sa_output = self.conv6(sa_conv)
    #
    #     feat2 = self.conv5c(x)
    #     sc_feat = self.sc(feat2)
    #     sc_conv = self.conv52(sc_feat)
    #     sc_output = self.conv7(sc_conv)
    #
    #     feat_sum = sa_conv + sc_conv
    #
    #     sasc_output = self.conv8(feat_sum)
    #
    #     output = [sasc_output]
    #     output.append(sa_output)
    #     output.append(sc_output)
    #     return tuple(output)

class DualAttention(nn.Module):
    """ Modified from the dualattention"""
    def __init__(self, in_dim):
        super().__init__()
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)

    def forward(self, x):
        """
        """
        pam_x = self.pam(x)
        cam_x = self.cam(x)
        return pam_x + cam_x
