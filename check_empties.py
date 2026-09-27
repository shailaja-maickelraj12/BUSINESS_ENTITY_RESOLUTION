import glob
import os
import pandas as pd

for path in sorted(glob.glob('student_resource/dataset/*/*.tsv')):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    empties = {}
    for col in df.columns:
        cnt = (df[col] == "").sum()
        pct = (cnt / len(df)) * 100
        empties[col] = f"{cnt:,} ({pct:.2f}%)"
    print(f"{os.path.basename(path)} | shape: {df.shape} | empties: {empties}")
