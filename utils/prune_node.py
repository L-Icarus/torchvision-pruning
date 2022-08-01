import torch
from torch import nn
from .prune_opset import PRUNE_CONV_OPTYPE, PRUNE_NORMALIZATION_OPTYPE, PRUNE_FC_OPTYPE
from .prune_tools import remove_dup_input, is_depthwise_conv


class PruneNode:
    def __init__(self, model_op, model_node, op_idx):
        """
        :param model_op: torch.nn.XX
        :param model_node: torch.fx.node.Node
        """
        # name实际上也是op在named_modules里的name
        self._name = model_node.target
        # idx是op在named_modules里的索引, 已经完成拓扑排序
        self._idx = op_idx
        # layer 对应 torch 算子
        self._op = model_op
        # node对应fx.GraphModel中的Node
        self._node = model_node
        # 当前prunenode的输入节点
        self._input_nodes = []
        # 接收当前prunenode输入的节点
        self._output_nodes = []
        # 与当前prunenode相同输入的节点
        self._same_input_nodes = []
        # 与当前prunenode相同输出的节点
        self._same_output_nodes = []
        # 输入通道是否被裁剪
        self.input_is_pruned = False
        # 输出通道是否被裁剪
        self.output_is_pruned = False
        # 输入通道裁剪方案(mask), True代表保留通道, is_same_io表示输入输出mask相同
        if isinstance(model_op, PRUNE_CONV_OPTYPE):
            self._input_mask = torch.ones(model_op.in_channels) == 1
            self._output_mask = torch.ones(model_op.out_channels) == 1
            self._is_same_io = is_depthwise_conv(model_op)
        elif isinstance(model_op, PRUNE_FC_OPTYPE):
            self._input_mask = torch.ones(model_op.in_features) == 1
            self._output_mask = torch.ones(model_op.out_features) == 1
            self._is_same_io = False
        elif isinstance(model_op, PRUNE_NORMALIZATION_OPTYPE):
            if hasattr(model_op, "num_features"):
                self._input_mask = torch.ones(model_op.num_features) == 1
                self._output_mask = torch.ones(model_op.num_features) == 1
            elif hasattr(model_op, "normalized_shape"):
                self._input_mask = torch.ones(model_op.normalized_shape[0]) == 1
                self._output_mask = torch.ones(model_op.normalized_shape[0]) == 1
            self._is_same_io = True
        else:
            raise TypeError
        # 临时对fc输入处理,vgg这种fc前特征图不为1×1
        self.rate = 1

    @property
    def name(self):
        return self._name

    @property
    def idx(self):
        return self._idx

    @property
    def op(self):
        return self._op

    @property
    def node(self):
        return self._node

    @property
    def input_nodes(self):
        return self._input_nodes

    @input_nodes.setter
    def input_nodes(self, new_input_nodes):
        assert isinstance(new_input_nodes, list), "Error: type of new_input_nodes is not List"
        self._input_nodes = new_input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def same_input_nodes(self):
        return self._same_input_nodes

    @property
    def same_output_nodes(self):
        return self._same_output_nodes

    @property
    def input_mask(self):
        return self._input_mask

    @property
    def output_mask(self):
        return self._output_mask

    def update_input_mask(self, mask=None, is_direct=True):
        # 如果输入已经裁剪,跳过
        if self.input_is_pruned:
            return True
        # 如果没有输入节点,标记为已裁剪
        if len(self._input_nodes) == 0:
            self.input_is_pruned = True
            return True
        # 如果是filter剪枝,输入是间接裁剪,输入节点的输出必须已经裁剪完成
        if not is_direct:
            input_nodes = remove_dup_input(self)
            for input_node in input_nodes:
                if not input_node.output_is_pruned:
                    return False
            mask = []
            for input_node in input_nodes:
                if len(mask) == 0:
                    mask = input_node.output_mask
                else:
                    mask = torch.concat((mask, input_node.output_mask))
            # 特殊处理fc
            if len(mask) != len(self._input_mask):
                l = mask.detach().cpu().sum()
                mask = torch.zeros(len(self._input_mask)) == 1
                mask[:l * self.rate] = True
            self._input_mask = mask
            self.input_is_pruned = True
        # 如果是channel剪枝,输入是直接裁剪,更新相同输入的节点
        else:
            # 特殊处理fc
            if len(mask) != len(self._input_mask):
                l = mask.detach().cpu().sum()
                mask = torch.zeros(len(self._input_mask)) == 1
                mask[:l * self.rate] = True
            self._input_mask = mask
            self.input_is_pruned = True
        for same_input_node in self._same_input_nodes:
            same_input_node.update_input_mask(mask)
        # 如果是depthwise conv和bn这类输入输出通道一致的节点,同时更新输入输出节点
        if self._is_same_io:
            self._output_mask = mask
            self.output_is_pruned = True
            for input_node in self._input_nodes:
                input_node.update_output_mask(mask)
            # 间接剪枝,因为后续节点可能没有准备好裁剪输入通道
            for output_node in self._output_nodes:
                output_node.update_input_mask(mask, False)
            for same_output_node in self._same_output_nodes:
                same_output_node.update_output_mask(mask)
        return True

    def update_output_mask(self, mask=None, is_direct=True):
        # 如果输出已经裁剪,跳过
        if self.output_is_pruned:
            return True
        # 如果没有输出节点,标记为已裁剪
        if len(self._output_nodes) == 0:
            self.output_is_pruned = True
            return True
        # 如果是channel剪枝,输出是间接裁剪,输出节点的输入必须已经裁剪完成
        if not is_direct:
            # 后续节点输入有一个已经被裁剪就可以间接裁剪当前节点输出(shufflenet除外)
            flag = False
            for output_node in self._output_nodes:
                if output_node.is_input_pruned:
                    flag = True
                    break
            if not flag:
                return False
            mask = []
            for output_node in self._output_nodes:
                if len(mask) == 0:
                    mask = output_node.input_mask
                else:
                    mask = torch.concat((mask, output_node.input_mask))
            self._output_mask = mask
            self.output_is_pruned = True
        # 如果是filter剪枝,输出是直接裁剪,更新相同输出的节点
        else:
            self._output_mask = mask
            self.output_is_pruned = True
        for same_output_node in self._same_output_nodes:
            same_output_node.update_output_mask(mask)
        # 如果是depthwise conv和bn这类输入输出通道一致的节点,同时更新输入输出节点
        if self._is_same_io:
            self._input_mask = mask
            self.input_is_pruned = True
            # 间接剪枝,因为后续节点可能没有准备好裁剪输入通道
            for output_node in self._output_nodes:
                output_node.update_input_mask(mask, False)
            for input_node in self._input_nodes:
                input_node.update_output_mask(mask)
            for same_input_node in self._same_input_nodes:
                same_input_node.update_input_mask(mask)
        return True


def transform_node_name(nodes):
    name_list = []
    for node in nodes:
        if isinstance(node, list):
            tmp = []
            for _node in node:
                tmp.append(_node.name)
            name_list.append(tmp)
        else:
            name_list.append(node.name)
    return name_list


def get_io_channels(p_node):
    if isinstance(p_node.op, PRUNE_CONV_OPTYPE):
        in_channels = p_node.op.in_channels
        out_channels = p_node.op.out_channels
    elif isinstance(p_node.op, PRUNE_FC_OPTYPE):
        in_channels = p_node.op.in_features
        out_channels = p_node.op.out_features
    else:
        if isinstance(p_node.op, nn.LayerNorm):
            in_channels = out_channels = p_node.op.normalized_shape[0]
        elif isinstance(p_node.op, nn.BatchNorm2d):
            in_channels = out_channels = p_node.op.num_features
    return in_channels, out_channels


def check_prune_same_input(p_node, other_nodes):
    p_ic, p_oc = get_io_channels(p_node)
    for node in other_nodes:
        n_ic, n_oc = get_io_channels(node)
        if p_ic != n_ic:
            return False
    return True


def check_prune_same_output(p_node, other_nodes):
    p_ic, p_oc = get_io_channels(p_node)
    for node in other_nodes:
        n_ic, n_oc = get_io_channels(node)
        if p_oc != n_oc:
            return False
    return True


def check_prune_input(p_node, other_nodes):
    p_ic, p_oc = get_io_channels(p_node)
    for node in other_nodes:
        n_ic, n_oc = get_io_channels(node)
        if p_ic != n_oc:
            return False
    return True


def check_prune_output(p_node, other_nodes):
    p_ic, p_oc = get_io_channels(p_node)
    for node in other_nodes:
        n_ic, n_oc = get_io_channels(node)
        if p_oc != n_ic:
            return False
    return True


def print_prune_node_info(p_node):
    print("**********")
    print(f"name: {p_node.name}")
    print(f"input nodes: {transform_node_name(p_node.input_nodes)} prune success: {check_prune_input(p_node, p_node.input_nodes)}")
    print(f"output nodes: {transform_node_name(p_node.output_nodes)} prune success: {check_prune_output(p_node, p_node.output_nodes)}")
    print(f"same input nodes: {transform_node_name(p_node.same_input_nodes)} prune success: {check_prune_same_input(p_node, p_node.same_input_nodes)}")
    print(f"same output nodes: {transform_node_name(p_node.same_output_nodes)} prune success: {check_prune_same_output(p_node, p_node.same_output_nodes)}")