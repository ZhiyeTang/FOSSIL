import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_variable(shape: list, initializer=nn.init.xavier_uniform_):
    return initializer(torch.empty(shape))


class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_heads: int,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(SelfAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.K = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.V = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.softmax = nn.Softmax().to(device)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        padding_mask: torch.Tensor,
        x: torch.Tensor,
    ):

        Qx = self.Q @ x
        Kx = self.K @ x
        Vx = self.V @ x
        Qh = torch.concat(torch.split(Qx, self.num_heads, dim=2), dim=0)
        Kh = torch.concat(torch.split(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.split(Vx, self.num_heads, dim=2), dim=0)

        output = torch.matmul(Qh, Kh.transpose(1, 2)) / Kh.shape[-1] ** 0.5
        tril = torch.tril(torch.ones_like(output[0, :, :]))
        casuality_mask = torch.tile(torch.unsqueeze(tril, 0), [
                                    output.shape[0], 1, 1])
        output = torch.where(torch.eq(casuality_mask, 0),
                             torch.ones_like(output)*(-2**32+1), output)

        key_mask = torch.tile(
            padding_mask, [self.num_heads, 1, self.Q.shape[1]]).transpose(1, 2)
        output = torch.where(torch.eq(key_mask, 0),
                             torch.ones_like(output)*(-2**32+1), output)

        output = self.softmax(output)

        query_mask = torch.tile(padding_mask, [self.num_heads, 1, x.shape[1]])
        output *= query_mask

        output = self.dropout(output)
        attention = torch.mean(torch.stack(torch.split(output[: ,-1], self.num_heads, dim=0), dim=0), dim=0)
        
        output = torch.matmul(output, Vh)
        output = torch.concat(torch.split(
            output, self.num_heads, dim=0), dim=-1)
        output = self.dropout(output)
        output += x

        return output, attention


class LocationBasedAttentionBlock(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_heads: int,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(LocationBasedAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q = nn.Parameter(
            get_variable([1, num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.K = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.V = nn.Parameter(
            get_variable([num_inputs, num_units]), 
            requires_grad=True, 
        ).to(device)
        self.softmax = nn.Softmax().to(device)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        padding_mask: torch.Tensor,
        x: torch.Tensor,
    ):

        Qx = self.Q @ x
        Kx = self.K @ x
        Vx = self.V @ x
        Qh = torch.concat(torch.split(Qx, self.num_heads, dim=2), dim=0)
        Kh = torch.concat(torch.split(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.split(Vx, self.num_heads, dim=2), dim=0)

        output = torch.matmul(Qh, Kh.transpose(1, 2))

        key_mask = torch.tile(
            padding_mask, [self.num_heads, 1, self.Q.shape[1]]).transpose(1, 2)
        output = torch.where(torch.eq(key_mask, 0),
                             torch.ones_like(output)*(-2**32+1), output)

        output = self.softmax(output)

        output = self.dropout(output)
        attention = torch.split(output[: ,-1], self.num_heads, dim=0)
        
        output = torch.matmul(output, Vh)
        output = torch.concat(torch.split(
            output, self.num_heads, dim=0), dim=-1)
        output = self.dropout(output)

        return output, attention


class FissaNetwork(nn.Module):
    def __init__(self):
        super(FissaNetwork).__init__()

        # Local Representation Parameters
        pass


class Algorithm:

    def __init__(self, config):
        pass
