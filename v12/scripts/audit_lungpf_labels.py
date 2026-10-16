import pandas as pd
from pathlib import Path
p=Path('data/benchmarking_data/lungpf')
a=pd.read_csv(p/'pathology.csv.gz')
print(a.groupby(['tma','sample']).size().to_string())
print(a.loc[a.tma.eq('TMA5')].head(10).to_string())
print(pd.read_csv(p/'GSM8505448_TILD028LA_transcripts.csv.gz',nrows=2).to_string())
