from abc import ABC, abstractmethod
from utils.prune_graph import PruneGraph


class Pruning(ABC):
    def __init__(self, model):
        self.model = model
        # 可剪枝层计算图
        self.prune_graph = PruneGraph(model)

    @abstractmethod
    def get_prune_mask(self, prune_node, prune_ratio=None):
        pass

    @abstractmethod
    def prune(self):
        pass
