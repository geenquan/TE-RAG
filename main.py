
import pandas as pd
from experiment.ablation import AblationExperiment
from experiment.comparison import Comparison

def main():
    data = pd.read_csv("your_dataset.csv")

    ablation_experiment = AblationExperiment(data)
    ablation_result = ablation_experiment.run()
    print("Ablation Experiment Results:", ablation_result)

    comparison = Comparison("model_method", "vector_search_method", "keyword_search_method")
    comparison_result = comparison.run_comparison("time")
    print("Comparison Time:", comparison_result)

if __name__ == "__main__":
    main()
