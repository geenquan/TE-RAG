
from abc import ABC, abstractmethod

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve_table(self, query_result):
        pass

    @abstractmethod
    def retrieve_field(self, table):
        pass
