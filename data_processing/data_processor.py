
import pandas as pd

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        self.data = self.data.dropna()  # 去除缺失值
        self.data = self.data.drop(columns=['id', 'is_deleted'])  # 去除不需要的字段
        return self.data
