from abc import ABC, abstractmethod


class FeatureHook(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, module, module_in, module_out):
        pass
