
class DeepSeekClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, query_text):
        result = {"table": "example_table", "columns": ["col1", "col2"]}
        return result
