from abc import ABC
import torch
from torch import nn


def get_parent_name(name):
    return name[:name.rfind('.')]


class MaskModel(ABC):
    # 用于在训练中软剪枝,为model中可裁剪op加上mask控制输入输出
    def __init__(self, model, prune_graph, mask_optype, op_prune_ratio, is_filterwise=True, need_train_mask=False):
        self.opnames = []
        # op名称对应mask
        self.opname2mask = {}
        # 以卷积为例param_name指的是conv.weight和conv.bias,对param修改实际上就是对卷积的权重和偏置修改
        self.opname2param = {}
        # mask作用于输入还是输出
        self.mask_on_filter = is_filterwise
        # 每个op的剪枝率
        self.op_prune_ratio = op_prune_ratio
        # 初始化mask
        for p_node in prune_graph.nodes:
            if isinstance(p_node.op, mask_optype):
                self.opnames.append(p_node.name)
                if is_filterwise:
                    self.opname2mask[p_node.name] = nn.Parameter(torch.ones(p_node.output_mask.size()),
                                                                 requires_grad=need_train_mask)
                else:
                    self.opname2mask[p_node.name] = nn.Parameter(torch.ones(p_node.input_mask.size()),
                                                                 requires_grad=need_train_mask)
        for name, param in model.named_parameters():
            self.opname2param.setdefault(get_parent_name(name), [])
            self.opname2param[get_parent_name(name)].append(param)

    def update_mask(self):
        pass

    def reset_mask(self, opname):
        mask = self.opname2mask[opname].data
        self.opname2mask[opname].data = torch.ones(mask.data.size())

    def get_mask(self, opname):
        return self.opname2mask[opname]

    def update_op_param(self, opname):
        mask = self.opname2mask[opname]
        for param in self.opname2param[opname]:
            param_dims = len(param.size())
            if self.mask_on_filter:
                # 卷积
                if param_dims == 4:
                    param.data *= mask.view(mask.size()[0], 1, 1, 1)
                # 全连接
                elif param_dims == 2:
                    param.data *= mask.view(mask.size()[0], 1)
                # 归一化
                else:
                    param.data *= mask
            else:
                # 卷积
                if param_dims == 4:
                    param.data *= mask.view(1, mask.size()[0], 1, 1)
                # 全连接
                elif param_dims == 2:
                    param.data *= mask.view(1, mask.size()[0])
                # 归一化
                else:
                    param.data *= mask

    def update_op_grad(self, opname):
        mask = self.opname2mask[opname]
        for param in self.opname2param[opname]:
            param_dims = len(param.size())
            if self.mask_on_filter:
                # 卷积
                if param_dims == 4:
                    param.grad.data *= mask.view(mask.size()[0], 1, 1, 1)
                # 全连接
                elif param_dims == 2:
                    param.grad.data *= mask.view(mask.size()[0], 1)
                # 归一化
                else:
                    param.grad.data *= mask
            else:
                # 卷积
                if param_dims == 4:
                    param.data *= mask.view(1, mask.size()[0], 1, 1)
                # 全连接
                elif param_dims == 2:
                    param.data *= mask.view(1, mask.size()[0])
                # 归一化
                else:
                    param.data *= mask

    def update_all_op_param(self):
        for opname in self.opnames:
            self.update_op_param(opname)

    def update_all_op_grad(self):
        for opname in self.opnames:
            self.update_op_grad(opname)
