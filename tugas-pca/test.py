import pandas as pd

df = pd.read_csv('First/first.csv',na_values='?')
print(df.isnull().any())