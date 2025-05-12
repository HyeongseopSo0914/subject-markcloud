import pandas as pd

df = pd.read_json('trademark_sample.json')
m_cnt = df.isnull().sum()
print(m_cnt)