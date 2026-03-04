
class ResourceComparator:
    def compare(self, method1_result, method2_result):
        return method1_result["memory"] - method2_result["memory"]
