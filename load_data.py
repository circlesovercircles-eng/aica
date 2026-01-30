# load_data.py
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"✅ Loaded shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("dataset/all_traffic_merged.csv")  # replace with actual path if needed
    print(df.head())
    print("Total rows:", df.shape[0])
