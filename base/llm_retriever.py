import pandas as pd
import requests
import json

from base.base_retriever import BaseRetriever


class DeepSeekRetriever(BaseRetriever):
    def __init__(self, table_csv, field_csv, api_key):
        # 读取CSV文件
        self.table_df = pd.read_csv(table_csv)
        self.field_df = pd.read_csv(field_csv)

        # 清理数据：将NaN或非字符串类型的数据转换为字符串
        self.table_df['table_desc'] = self.table_df['table_desc'].apply(lambda x: str(x) if isinstance(x, str) else '')
        self.field_df['field_name_desc'] = self.field_df['field_name_desc'].apply(
            lambda x: str(x) if isinstance(x, str) else '')

        # 存储API Key
        self.api_key = api_key

        # 构建数据
        self.table_data = self.table_df.to_dict(orient="records")
        self.field_data = self.field_df.to_dict(orient="records")

    def deepseek_inference(self, query, data):
        # DeepSeek API URL（替换为你的API URL）
        url = "https://api.deepseek.com/v1/query"

        headers = {
            'Authorization': f'Bearer {self.api_key}',  # 使用API Key进行身份验证
            'Content-Type': 'application/json',
        }

        payload = {
            "query": query,
            "data": data
        }

        # 发送POST请求到DeepSeek API进行推理
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # 检查请求是否成功
        if response.status_code == 200:
            return response.json()  # 返回JSON格式的结果
        else:
            raise Exception(f"DeepSeek API请求失败，状态码：{response.status_code}, 错误信息：{response.text}")

    def retrieve_table(self, query_result):
        # 调用DeepSeek模型进行查询，返回与查询相关的表格
        results = self.deepseek_inference(query_result, self.table_data)

        # 根据返回的相关表格索引返回结果
        return pd.DataFrame(results)

    def retrieve_field(self, table):
        # 根据表名过滤字段数据
        table_fields = self.field_df[self.field_df['table'] == table]

        # 调用DeepSeek模型进行字段查询
        results = self.deepseek_inference(table, table_fields.to_dict(orient="records"))

        # 根据返回的相关字段索引返回结果
        return pd.DataFrame(results)


# 测试代码
def main():
    # 设置DeepSeek的API Key
    api_key = "sk-e7ac43eb20474155acd2d1f435164acc"  # 替换为实际的API Key

    # 创建检索实例
    retriever = DeepSeekRetriever('/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv', '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv',
                                  api_key)

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