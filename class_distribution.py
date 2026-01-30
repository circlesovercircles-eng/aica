import pandas as pd

def get_class_distribution(df, label_col='Label'):
    class_counts = df[label_col].value_counts().sort_index()
    print("📊 Current class distribution:\n", class_counts)
    return class_counts

if __name__ == "__main__":
    df = pd.read_csv("dataset/all_traffic_merged.csv")
    get_class_distribution(df)
