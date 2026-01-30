import pandas as pd

# Paste your earlier mapping here (from when you printed dict(zip(le.classes_, le.transform(le.classes_))))
label_mapping = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenEye',
    4: 'DoS Hulk',
    5: 'DoS Slowhttptest',
    6: 'DoS slowloris',
    7: 'FTP-Patator',
    8: 'Heartbleed',
    9: 'Infiltration',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack - Brute Force',
    13: 'Web Attack - Sql Injection',
    14: 'Web Attack - XSS'
}
df = pd.read_csv("dataset/preprocessed_traffic.csv")

# Now use it
counts = df['Label_encoded'].value_counts().sort_values(ascending=False)

dist_df = pd.DataFrame({
    'Encoded': counts.index,
    'Class': [label_mapping.get(code, f'Unknown_{code}') for code in counts.index],
    'Count': counts.values,
    'Percentage': (counts / len(df) * 100).round(2)
})

dist_df['Cumulative %'] = dist_df['Percentage'].cumsum().round(2)

print(dist_df.to_string(index=False))