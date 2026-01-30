import pandas as pd
import os
files = [
    r"dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    r"dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    r"dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    r"dataset/Monday-WorkingHours.pcap_ISCX.csv",
    r"dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    r"dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    r"dataset/Tuesday-WorkingHours.pcap_ISCX.csv",
    r"dataset/Wednesday-workingHours.pcap_ISCX.csv",
]

print(f"Will merge {len(files)} files\n")

dfs = []
for file in files:
    print(f"Reading {os.path.basename(file)} ...")
    df = pd.read_csv(file, encoding='latin1', low_memory=False)
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(' ', '_')
    dfs.append(df)

print("\nConcatenating...")
merged = pd.concat(dfs, ignore_index=True, sort=False)

output = "dataset/all_traffic_merged.csv"
merged.to_csv(output, index=False)
print(f"Saved → {output}  (rows: {len(merged):,})")