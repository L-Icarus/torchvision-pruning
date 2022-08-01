# https://arxiv.org/pdf/1607.03250.pdf
import math
import torch
from torch.nn import functional as F
from .base_pruning import Pruning
from utils.hook import FeatureHook
from utils.prune_opset import PRUNE_CONV_OPTYPE, PRUNE_FC_OPTYPE
from utils.prune_tools import prune_conv, prune_fc, prune_norm, is_group_conv, get_lcm, find_next_conv_list, \
    is_depthwise_conv


class APoZHook(FeatureHook):
    def __init__(self):
        super().__init__()
        self.feature_result = torch.tensor(0.)
        self.total = torch.tensor(0.)

    def __call__(self, module, module_in, module_out):
        feature_after_relu = F.relu(module_out, inplace=False)
        # 根据所有输入样本求每个filter的APoZ
        zero_cnt = (feature_after_relu == 0).sum(dim=(0, 2, 3))
        feature_cnt = (feature_after_relu >= 0).sum(dim=(0, 2, 3))
        apoz = zero_cnt / feature_cnt
        self.feature_result = self.feature_result * self.total + apoz
        self.total += module_out.shape[0]
        self.feature_result /= self.total

    def get_feature(self):
        return self.feature_result

    def reset(self):
        self.feature_result = torch.tensor(0.)
        self.total = torch.tensor(0.)


class APoZPruner(Pruning):
    """
    APoZ pruning is filter pruning algorithm
    APoZ需要先为每个卷积添加hook,然后在训练集上跑一些iter来提取特征进行剪枝
    """

    def __init__(self, model, op_prune_ratio=0.5):
        """
        :param model: fx.GraphModel
        :param op_prune_ratio: dict of op and prune ratio
        """
        super().__init__(model)
        # 索引对应的剪枝率, 将全局剪枝率转化为每层剪枝率
        if isinstance(op_prune_ratio, float):
            # self.op_prune_ratio = dict(zip(self.prune_op_name, [op_prune_ratio] * len(self.prune_op_name)))
            # debug
            import random
            self.op_prune_ratio = dict(zip(self.prune_graph.node_names,
                                           [random.random() for i in range(len(self.prune_graph.node_names))]))
        else:
            assert len(op_prune_ratio) == len(self.prune_graph.node_names), \
                "number of op_prune_ratio is not match number of prune_op_name"
            self.op_prune_ratio = op_prune_ratio

        # 保存每一个卷积的rank hook
        self._hook_dict = {}
        # 管理hook的handler
        self._handler_list = []

    def insert_hook(self):
        for prune_node in self.prune_graph.nodes:
            if isinstance(prune_node.op, PRUNE_CONV_OPTYPE):
                hook = APoZHook()
                self._hook_dict[prune_node.name] = hook
                handler = prune_node.op.register_forward_hook(hook)
                self._handler_list.append(handler)

    def remove_hook(self):
        for handler in self._handler_list:
            handler.remove()

    def get_prune_mask(self, prune_node, prune_ratio=None):
        channels = prune_node.op.out_channels
        save_channels = math.ceil(channels * (1 - prune_ratio))
        # 注意剪枝后对分组卷积输入输出的影响
        # 如果自己是分组卷积
        if is_group_conv(prune_node.op):
            self_groups = prune_node.op.groups
            if save_channels % self_groups != 0:
                save_channels += self_groups - save_channels % self_groups
        else:
            self_groups = 1
        for next_prune_node in prune_node.output_nodes:
            # 如果后续节点是分组卷积但不是深度卷积
            if isinstance(next_prune_node.op, PRUNE_CONV_OPTYPE) and is_group_conv(
                    next_prune_node.op) and not is_depthwise_conv(next_prune_node.op):
                groups = next_prune_node.op.groups
                # 如果当前卷积和下一个卷积都是分组卷积, 求group的最小公倍数
                self_groups = get_lcm(self_groups, groups)
            else:
                # 如果后继节点不是分组卷积, 先搜索一下
                next_p_node_list = find_next_conv_list(next_prune_node)
                for next_p_node in next_p_node_list:
                    if isinstance(next_p_node.op, PRUNE_CONV_OPTYPE) and is_group_conv(
                            next_p_node.op) and not is_depthwise_conv(next_p_node.op):
                        groups = next_p_node.op.groups
                        # 如果当前卷积和下一个卷积都是分组卷积, 求group的最小公倍数
                        self_groups = get_lcm(self_groups, groups)
        if save_channels % self_groups != 0:
            save_channels += self_groups - save_channels % self_groups
            if save_channels > channels:
                save_channels = channels
        # 保留APoZ低的filter
        _, index = torch.topk(self._hook_dict[prune_node.name].get_feature(), k=save_channels, largest=False)
        mask = torch.zeros(channels) == 1
        mask[index] = True
        return mask

    def prune(self):
        # 获取每个op的剪枝方案
        for prune_node in self.prune_graph.nodes:
            prune_node.update_input_mask(is_direct=False)
            if isinstance(prune_node.op, PRUNE_CONV_OPTYPE):
                prune_node.update_output_mask(self.get_prune_mask(prune_node, self.op_prune_ratio[prune_node.name]))
        # 剪枝过程
        for prune_node in self.prune_graph.nodes:
            if isinstance(prune_node.op, PRUNE_CONV_OPTYPE):
                prune_conv(prune_node.op, prune_node.input_mask, True)
                if not is_depthwise_conv(prune_node.op):
                    prune_conv(prune_node.op, prune_node.output_mask, False)
            elif isinstance(prune_node.op, PRUNE_FC_OPTYPE):
                prune_fc(prune_node.op, prune_node.input_mask, True)
                prune_fc(prune_node.op, prune_node.output_mask, False)
            else:
                prune_norm(prune_node.op, prune_node.input_mask)
        # 移除hook
        self.remove_hook()
