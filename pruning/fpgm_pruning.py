# https://arxiv.org/pdf/1811.00250.pdf
import math
import torch
import numpy as np
from scipy.spatial import distance
from .base_pruning import Pruning
from utils.prune_wrapper import MaskModel
from utils.prune_opset import PRUNE_CONV_OPTYPE, PRUNE_FC_OPTYPE
from utils.prune_tools import prune_conv, prune_fc, prune_norm, is_group_conv, get_lcm, find_next_conv_list, \
    is_depthwise_conv


class FPGMMaskModel(MaskModel):
    def __init__(self, model, prune_graph, mask_optype, op_prune_ratio, is_filterwise=True, need_train_mask=False):
        super().__init__(model, prune_graph, mask_optype, op_prune_ratio, is_filterwise,
                                            need_train_mask)

    # https://github.com/he-y/filter-pruning-geometric-median/blob/76f0ffefbdba7335251bfbafaadce49a2f08d2b9/pruning_cifar10.py#L491 get_filter_similar
    def update_mask(self, opname, prune_channel_number=-1):
        self.reset_mask(opname)
        # 这里是tensor不是parameter
        mask = self.opname2mask[opname].data
        if prune_channel_number != -1:
            similar_pruned_num = prune_channel_number
        else:
            similar_pruned_num = int(mask.size()[0] * self.op_prune_ratio[opname])
        # opname的第一个参数一定是weight
        weight_vec = self.opname2param[opname][0].data.view(mask.size()[0], -1)
        norm = torch.norm(weight_vec, 2, 1)
        norm_np = norm.cpu().numpy()
        filter_large_index = norm_np.argsort()

        # distance using numpy function
        indices = torch.LongTensor(filter_large_index)
        weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
        # for euclidean distance
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
        similar_sum = np.sum(np.abs(similar_matrix), axis=0)

        # for distance similar: get the filter index with largest similarity == small distance
        similar_large_index = similar_sum.argsort()[similar_pruned_num:]
        similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
        similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
        for idx in similar_index_for_filter:
            mask[idx] = 0
        self.opname2mask[opname].data = mask
        return mask

    def update_all_mask(self):
        for opname in self.opnames:
            self.update_mask(opname)


class FPGMPruner(Pruning):
    """
    fpgm pruning is filter pruning algorithm
    每个batch调用update_all_op_grad更新梯度,每个epoch先调用update_all_mask,再调用update_all_op_param
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
        self.wrapped_model = FPGMMaskModel(self.model, self.prune_graph, PRUNE_CONV_OPTYPE, self.op_prune_ratio,
                                           is_filterwise=True, need_train_mask=False)

    def get_prune_mask(self, prune_node, prune_ratio=None):
        # 考虑分组卷积的情况要调整裁剪比例
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
                # 如果后继节点不是分组卷积（norm）, 先搜索一下
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
        # save_channels没有变化直接获取软掩码作为剪枝方案
        if save_channels == math.ceil(channels * (1 - prune_ratio)):
            mask = self.wrapped_model.update_mask(prune_node.name)
            # param可以不需要更新,因为是结构化剪枝
            self.wrapped_model.update_op_param(prune_node.name)
        else:
            mask = self.wrapped_model.update_mask(prune_node.name, channels - save_channels)
            # param可以不需要更新,因为是结构化剪枝
            self.wrapped_model.update_op_param(prune_node.name)
        # mask布尔化
        mask = mask == 1
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
