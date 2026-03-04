import pandas as pd
from collections import Counter
import math
import numpy as np

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve_table(self, query_result):
        pass

    @abstractmethod
    def retrieve_field(self, table):
        pass


class BM25Retriever(BaseRetriever):
    def __init__(self, table_csv, field_csv):
        # 读取CSV数据
        self.table_df = pd.read_csv(table_csv)
        self.field_df = pd.read_csv(field_csv)

        # 清理数据：将NaN或非字符串类型的数据转换为字符串
        self.table_df['table_desc'] = self.table_df['table_desc'].apply(lambda x: str(x) if isinstance(x, str) else '')
        self.field_df['field_name_desc'] = self.field_df['field_name_desc'].apply(
            lambda x: str(x) if isinstance(x, str) else '')

        # 创建倒排索引
        self.table_inverted_index = self.create_inverted_index(self.table_df['table_desc'])
        self.field_inverted_index = self.create_inverted_index(self.field_df['field_name_desc'])

        # 计算文档频率
        self.table_doc_freq = self.compute_doc_freq(self.table_inverted_index)
        self.field_doc_freq = self.compute_doc_freq(self.field_inverted_index)

        # 计算文档总数
        self.total_table_docs = len(self.table_df)
        self.total_field_docs = len(self.field_df)

        # 文档长度（词汇数量）
        self.table_doc_lengths = self.compute_doc_lengths(self.table_df['table_desc'])
        self.field_doc_lengths = self.compute_doc_lengths(self.field_df['field_name_desc'])

        # 平均文档长度
        self.avg_table_doc_length = np.mean(self.table_doc_lengths)
        self.avg_field_doc_length = np.mean(self.field_doc_lengths)

        # 设置BM25的调节参数
        self.k1 = 1.5
        self.b = 0.75

    def create_inverted_index(self, texts):
        inverted_index = {}
        for idx, text in enumerate(texts):
            words = text.split()  # 基于空格分词，实际使用时可以根据需要使用更复杂的分词方法
            word_count = Counter(words)
            for word, count in word_count.items():
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append((idx, count))
        return inverted_index

    def compute_doc_freq(self, inverted_index):
        doc_freq = {}
        for word, postings in inverted_index.items():
            doc_freq[word] = len(set([posting[0] for posting in postings]))  # 不重复计算文档频率
        return doc_freq

    def compute_doc_lengths(self, texts):
        return [len(text.split()) for text in texts]

    def compute_idf(self, term, total_docs, doc_freq):
        df = doc_freq.get(term, 0)
        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def compute_bm25_score(self, query_terms, doc_idx, inverted_index, doc_lengths, avg_doc_length, doc_freq):
        score = 0
        doc_length = doc_lengths[doc_idx]
        for term in query_terms:
            if term in inverted_index:
                postings = inverted_index[term]
                term_freq = next((count for idx, count in postings if idx == doc_idx), 0)
                idf = self.compute_idf(term, len(doc_lengths), doc_freq)
                score += idf * ((term_freq * (self.k1 + 1)) / (
                            term_freq + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))))
        return score

    def retrieve_table(self, query_result):
        query_terms = query_result.split()
        scores = []

        # 对每个表，计算其与查询的BM25得分
        for doc_idx in range(self.total_table_docs):
            score = self.compute_bm25_score(query_terms, doc_idx, self.table_inverted_index, self.table_doc_lengths,
                                            self.avg_table_doc_length, self.table_doc_freq)
            scores.append((doc_idx, score))

        # 按照得分排序返回前5个表
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_5_idx = [score[0] for score in sorted_scores[:5]]
        return self.table_df.iloc[top_5_idx]

    def retrieve_field(self, table):
        # 根据表名过滤字段数据
        table_fields = self.field_df[self.field_df['table'] == table]

        # 将字段描述转化为查询词
        query_terms = " ".join(table_fields['field_name_desc']).split()

        # 对每个字段，计算其与查询的BM25得分
        scores = []
        for doc_idx in range(len(table_fields)):
            score = self.compute_bm25_score(query_terms, doc_idx, self.field_inverted_index, self.field_doc_lengths,
                                            self.avg_field_doc_length, self.field_doc_freq)
            scores.append((doc_idx, score))

        # 按照得分排序返回前5个字段
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_5_idx = [score[0] for score in sorted_scores[:5]]
        return table_fields.iloc[top_5_idx]


# 测试代码
def main():
    # 创建检索实例
    retriever = BM25Retriever('/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv', '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv')

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