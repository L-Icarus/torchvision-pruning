import operator
from .prune_opset import ALL_PRUNE_OPTYPE


# 判断一个fx node是否是module类型, module是模型中在self.__init__中声明的部分, 可剪枝op都是module类型
def is_call_module(fx_node):
    return fx_node.op == "call_module"


class FxNodeManager:
    # 管理fx.node.Node
    def __init__(self, graph_model, opname2op):
        # Node name对应的输入Node name
        self._name2input_name = {}
        self._name2output_name = {}
        self._name2fxnode = {}
        # opname实际上是named_modules里的name, 也是fx node中的target
        self._opname2fxnode = {}
        self._opname2op = opname2op
        self._initialize(graph_model)

    @property
    def name2input_name(self):
        return self._name2input_name

    @property
    def name2output_name(self):
        return self._name2output_name

    @property
    def name2fxnode(self):
        return self._name2fxnode

    @property
    def opname2fxnode(self):
        return self._opname2fxnode

    # 得到GraphModel中所有node的输入输出节点
    def _initialize(self, graph_model):
        for node in graph_model.graph.nodes:
            self._name2input_name.setdefault(node.name, [])
            self._name2output_name.setdefault(node.name, [])
            self._name2fxnode[node.name] = node
            # 这里只加入可剪枝的node(maxpool这类不算), 便于将nn.Module与fx.GraphModule关联
            if is_call_module(node) and isinstance(self._opname2op[node.target], ALL_PRUNE_OPTYPE):
                self._opname2fxnode[node.target] = node
            for input_node in node.all_input_nodes:
                self._name2input_name[node.name].append(input_node.name)
                self._name2output_name[input_node.name].append(node.name)

    # 根据可剪枝op的fx node找到对应输入op
    def get_fxnode_input_modules(self, fx_node_name):
        input_names = self._name2input_name[fx_node_name]
        input_modules = []
        for inp_name in input_names:
            inp_fx_node = self._name2fxnode[inp_name]
            if is_call_module(inp_fx_node) and isinstance(self._opname2op[inp_fx_node.target], ALL_PRUNE_OPTYPE):
                # 获取module(op)对应的名称
                input_modules.append(inp_fx_node.target)
            else:
                input_modules.extend(self.get_fxnode_input_modules(inp_fx_node.name))
        return input_modules

    # 根据可剪枝op的fx node找到对应输出op
    def get_fxnode_output_modules(self, fx_node_name):
        output_names = self._name2output_name[fx_node_name]
        output_modules = []
        for outp_name in output_names:
            outp_fx_node = self._name2fxnode[outp_name]
            if is_call_module(outp_fx_node) and isinstance(self._opname2op[outp_fx_node.target], ALL_PRUNE_OPTYPE):
                # 获取module(op)对应的名称
                output_modules.append(outp_fx_node.target)
            else:
                output_modules.extend(self.get_fxnode_output_modules(outp_name))
        return output_modules

    # 找到shortcut, 网络中对多个op进行add和mul操作
    def get_fxnode_shortcut_modules(self, fx_node_name):
        output_names = self._name2output_name[fx_node_name]
        shortcut_modules = []
        for outp_name in output_names:
            outp_fx_node = self._name2fxnode[outp_name]
            # 跳过可剪枝模块
            if is_call_module(outp_fx_node) and isinstance(self._opname2op[outp_fx_node.target], ALL_PRUNE_OPTYPE):
                continue
            else:
                sub_shortcut_modules = self.get_fxnode_shortcut_modules(outp_name)
                for sub in sub_shortcut_modules:
                    if sub not in shortcut_modules:
                        shortcut_modules.append(sub)
        # 如果自己是add mul, 把输入加入shortcut
        fx_node = self._name2fxnode[fx_node_name]
        if not is_call_module(fx_node) and fx_node.target in [operator.add, operator.mul]:
            sub_shortcut_modules = self.get_fxnode_input_modules(fx_node_name)
            for sub in sub_shortcut_modules:
                if sub not in shortcut_modules:
                    shortcut_modules.append(sub)
        # 把自己从shortcut中删除
        if fx_node.target in shortcut_modules:
            shortcut_modules.remove(fx_node.target)
        return shortcut_modules

