
from abc import ABC, abstractmethod

class BaseComparator(ABC):
    @abstractmethod
    def compare_time(self, method1_result, method2_result):
        pass

    @abstractmethod
    def compare_memory(self, method1_result, method2_result):
        pass
