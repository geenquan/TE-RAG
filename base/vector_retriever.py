import pandas as pd
from abc import ABC, abstractmethod
import faiss
import numpy as np

from base.base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    def __init__(self, table_csv, field_csv):
        # 读取CSV数据
        self.table_df = pd.read_csv(table_csv)
        self.field_df = pd.read_csv(field_csv)

        # 创建两个向量索引
        self.table_index = self.create_index(self.table_df[['table', 'table_desc']].values)
        self.field_index = self.create_index(self.field_df[['field_name', 'field_name_desc']].values)

    def create_index(self, data):
        # 假设每个字段是一个2D向量，我们需要用一些方法（如TF-IDF，BERT等）将文本转化为向量
        # 这里我们用随机生成的向量作为示例
        vectors = np.random.rand(len(data), 128).astype('float32')  # 随机生成向量（替换为真实向量化方法）

        # 使用FAISS创建索引
        index = faiss.IndexFlatL2(128)  # 使用L2距离来计算相似度
        index.add(vectors)
        return index

    def retrieve_table(self, query_result):
        # 假设查询结果是一个文本，我们需要转换为向量
        query_vector = np.random.rand(1, 128).astype('float32')  # 随机生成查询向量（替换为真实向量化方法）

        # 在表级别索引中检索
        D, I = self.table_index.search(query_vector, 5)  # 返回5个最相似的结果
        return self.table_df.iloc[I[0]]  # 返回表级别的最相似条目

    def retrieve_field(self, table):
        # 根据表名过滤字段数据
        table_fields = self.field_df[self.field_df['table'] == table]

        # 将字段描述转化为向量并检索
        query_vector = np.random.rand(1, 128).astype('float32')  # 随机生成查询向量（替换为真实向量化方法）
        vectors = np.random.rand(len(table_fields), 128).astype('float32')  # 随机生成字段的向量（替换为真实向量化方法）

        # 创建一个FAISS索引
        field_index = faiss.IndexFlatL2(128)
        field_index.add(vectors)

        D, I = field_index.search(query_vector, 5)  # 返回5个最相似的字段
        return table_fields.iloc[I[0]]  # 返回字段级别的最相似条目


# 测试代码
def main():
    # 创建检索实例
    retriever = VectorRetriever('/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv', '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv')

    # 测试表级别的检索
    query_result_table = "基层数据服务供电所看板汇总报表(月)"  # 假设查询结果为表的描述
    print("检索到的表：")
    print(retriever.retrieve_table(query_result_table))

    # 测试字段级别的检索
    table_name = "ads_itg_jcsjfw_gdskb_report_mf"
    print("\n检索到的字段：")
    print(retriever.retrieve_field(table_name))


if __name__ == "__main__":
    main()