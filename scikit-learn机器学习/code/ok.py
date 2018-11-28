import pandas as pd

data = {'col1': {'wr': 30, 'wr1': 20, 'wr2': 50},
        'col2': {'wr': 100, 'wr1': 90, 'wr2': 80},
        'col3': {'wr': 91, 'wr1': 90, 'wr2': 80}}
df = pd.DataFrame(data)
print(df)
df = df.loc['wr'][df.loc['wr'] > 90].index.values
print(df)
