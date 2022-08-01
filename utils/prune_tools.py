import math
import torch
from torch import nn
from .prune_opset import PRUNE_CONV_OPTYPE
from torchvision.models.convnext import LayerNorm2d


def is_group_conv(conv):
    return conv.groups != 1


def is_depthwise_conv(conv):
    return conv.in_channels == conv.out_channels and conv.groups == conv.in_channels


def prune_conv(conv, mask, is_input=False):
    if is_input:
        if is_group_conv(conv):
            prune_group_conv(conv, mask)
        else:
            conv.weight = nn.Parameter(conv.weight[:, mask])
            conv.in_channels = mask.sum().item()
    else:
        if is_depthwise_conv(conv):
            prune_group_conv(conv, mask)
        else:
            conv.weight = nn.Parameter(conv.weight[mask])
            if conv.bias is not None:
                conv.bias = nn.Parameter(conv.bias[mask])
            conv.out_channels = mask.sum().item()
    return conv


def prune_group_conv(conv, mask):
    # 只有通道剪枝与普通卷积有区别, filter剪枝与普通卷积一致
    new_mask = torch.zeros(conv.in_channels // conv.groups) == 1
    save_channels = mask.sum().item()
    groups_channels = save_channels // conv.groups
    # 分组卷积输入需要单独处理, 输出不需要
    if not is_depthwise_conv(conv):
        # 在输入通道求l1
        _, index = torch.topk(torch.sum(torch.abs(conv.weight), dim=(0, 2, 3)), k=groups_channels)
        new_mask[index] = True
        conv.weight = nn.Parameter(conv.weight[:, new_mask])
        conv.in_channels = save_channels
    else:
        conv.weight = nn.Parameter(conv.weight[mask])
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[mask])
        conv.in_channels = conv.out_channels = conv.groups = save_channels
    return conv


def prune_norm(norm, mask=None, threshold=0, l_bound=0.01):
    if mask is not None:
        norm.weight = nn.Parameter(norm.weight[mask])
        norm.bias = nn.Parameter(norm.bias[mask])
        # batch norm
        if hasattr(norm, 'running_mean'):
            norm.register_buffer('running_mean', norm.running_mean[mask])
            norm.register_buffer('running_var', norm.running_var[mask])
        # layer norm
        if hasattr(norm, 'normalized_shape'):
            norm.normalized_shape = (mask.sum().item(), )
        norm.num_features = mask.sum().item()
        return norm
    else:
        out_mask = norm.weight.data.abs() > threshold
        if out_mask.sum().item() < math.ceil(norm.num_features * l_bound):
            out_mask[:] = False
            weights = norm.weight.data.abs().sum(dim=(1, 2, 3)).detach().cpu().numpy()
            top_k = math.ceil(norm.num_features * l_bound)
            top_k_idx = weights.argsort()[::-1][0:top_k]
            out_mask[top_k_idx.copy()] = True
        norm.weight = nn.Parameter(norm.weight[out_mask])
        norm.bias = nn.Parameter(norm.bias[out_mask])
        norm.register_buffer('running_mean', norm.running_mean[out_mask])
        norm.register_buffer('running_var', norm.running_var[out_mask])
        norm.num_features = out_mask.sum().item()
        return norm, out_mask


def prune_fc(fc, mask, is_input=False):
    if is_input:
        fc.weight = nn.Parameter(fc.weight[:, mask])
        fc.in_features = mask.sum().item()
    else:
        fc.weight = nn.Parameter(fc.weight[mask, :])
        fc.out_features = mask.sum().item()
        if fc.bias is not None:
            fc.bias = nn.Parameter(fc.bias[mask])
    return fc


# 最小公倍数
def get_lcm(x, y):
    if x > y:
        greater = x
    else:
        greater = y
    while True:
        if greater % x == 0 and greater % y == 0:
            lcm = greater
            break
        greater += 1
    return lcm


# 可能后续有多个卷积
def find_next_conv_list(p_node):
    next_conv_list = []
    for output_p_node in p_node.output_nodes:
        if isinstance(output_p_node.op, PRUNE_CONV_OPTYPE):
            next_conv_list.append(output_p_node)
        else:
            next_conv_list.extend(find_next_conv_list(output_p_node))
    return next_conv_list


# 排除输入中的相同输出节点
def remove_dup_input(p_node):
    all_inputs = p_node.input_nodes
    n = len(all_inputs)
    new_inputs = []
    for i in range(n):
        flag = True
        for j in range(i + 1, n):
            if all_inputs[i] in all_inputs[j].same_output_nodes:
                flag = False
                break
        if flag:
            new_inputs.append(all_inputs[i])
    return new_inputs
