from .prune_opset import ALL_PRUNE_OPTYPE, PRUNE_FC_OPTYPE
from .prune_tools import remove_dup_input
from .fx_node_manager import FxNodeManager
from .prune_node import PruneNode


def get_name_op_map(model):
    """
    :param model: torch.Model, fx.GraphModel
    :return: mapping of name&op and name&idx
    """
    name2op = {}
    name2idx = {}
    for idx, m in enumerate(model.named_modules()):
        name2op[m[0]] = m[1]
        name2idx[m[0]] = idx
    return name2op, name2idx


def find_prune_layer(model):
    layer_name = []
    for idx, m in enumerate(model.named_modules()):
        # norm layer也算可剪枝op
        if isinstance(m[1], ALL_PRUNE_OPTYPE):
            layer_name.append(m[0])
    return layer_name


def find_same_input_layer(all_opnames, all_inputs):
    same_input_layer = {}
    for opname_a in all_opnames:
        same_input_layer.setdefault(opname_a, [])
        for opname_b in all_opnames:
            if opname_a != opname_b and all_inputs[opname_a] == all_inputs[opname_b]:
                same_input_layer[opname_a].append(opname_b)
    return same_input_layer


def find_same_output_layer(all_opnames, all_outputs):
    same_output_layer = {}
    for opname_a in all_opnames:
        same_output_layer.setdefault(opname_a, [])
        for opname_b in all_opnames:
            if opname_a != opname_b and all_outputs[opname_a] == all_outputs[opname_b]:
                same_output_layer[opname_a].append(opname_b)
    return same_output_layer


def check_finish_prune(p_node):
    return p_node.input_is_pruned and p_node.output_is_pruned


class PruneGraph:
    def __init__(self, graph_model):
        # 所有可剪枝op的PruneNode封装
        self._p_nodes = []
        # 所有可剪枝op的name
        self._p_names = []
        self._initialize(graph_model)

    @property
    def nodes(self):
        return self._p_nodes

    @property
    def node_names(self):
        return self._p_names

    def _initialize(self, graph_model):
        opname2op, opname2idx = get_name_op_map(graph_model)
        fx_node_mgr = FxNodeManager(graph_model, opname2op)
        # 所有可裁剪op name
        prune_opname = find_prune_layer(graph_model)
        opname2input_opname = {}
        opname2output_opname = {}
        opname2same_output_opname = {}
        for opname in prune_opname:
            fx_node = fx_node_mgr.opname2fxnode[opname]
            # 按在name_modules出现顺序排序, 便于通道对齐
            opname2input_opname[opname] = sorted(fx_node_mgr.get_fxnode_input_modules(fx_node.name), key=lambda x: opname2idx[x])
            opname2output_opname[opname] = fx_node_mgr.get_fxnode_output_modules(fx_node.name)
            opname2same_output_opname[opname] = fx_node_mgr.get_fxnode_shortcut_modules(fx_node.name)
        opname2same_input_opname = find_same_input_layer(prune_opname, opname2input_opname)
        # 创建PruneGraph
        opname2p_node = {}
        # 创建PruneNode, 暂时不处理输入输出节点
        for opname in prune_opname:
            p_node = PruneNode(opname2op[opname], fx_node_mgr.opname2fxnode[opname], opname2idx[opname])
            opname2p_node[opname] = p_node
            self.append(p_node)
        # 添加输入输出节点
        for opname in prune_opname:
            p_node = opname2p_node[opname]
            # 输入节点
            for input_opname in opname2input_opname[opname]:
                p_node.input_nodes.append(opname2p_node[input_opname])
            # fc特殊处理
            in_channels = 0
            if isinstance(p_node.op, PRUNE_FC_OPTYPE):
                input_nodes = remove_dup_input(p_node)
                for input_node in input_nodes:
                    in_channels += len(input_node.output_mask)
                if in_channels != len(p_node.input_mask):
                    p_node.rate = len(p_node.input_mask) // in_channels
            # 输出节点
            for output_name in opname2output_opname[opname]:
                p_node.output_nodes.append(opname2p_node[output_name])
            # 相同输入节点
            if opname in opname2same_input_opname:
                for same_input_opname in opname2same_input_opname[opname]:
                    p_node.same_input_nodes.append(opname2p_node[same_input_opname])
            # 相同输出节点
            if opname in opname2same_output_opname:
                for same_output_name in opname2same_output_opname[opname]:
                    p_node.same_output_nodes.append(opname2p_node[same_output_name])

    def append(self, node):
        assert isinstance(node, PruneNode), "Error: Add nodes that are not of PruneNode "
        self._p_nodes.append(node)
        self._p_names.append(node.name)

    def finish_pruning(self):
        for p_node in self._p_nodes:
            if not check_finish_prune(p_node):
                return False
        return True
