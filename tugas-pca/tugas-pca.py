import knn_impute as ki
import pandas as pd

df = pd.read_csv('First/first.csv')
new_data = ki.knn_impute(target=df['trestbps'], attributes=df.drop(['trestbps', 'binaryClass'], 1),
                                    aggregation_method="mean", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(trestbps = new_data)
new_data = ki.knn_impute(target=df['chol'], attributes=df.drop(['chol', 'binaryClass'], 1),
                                    aggregation_method="mean", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(chol = new_data)
new_data = ki.knn_impute(target=df['fbs'], attributes=df.drop(['fbs', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(fbs = new_data)
new_data = ki.knn_impute(target=df['restecg'], attributes=df.drop(['restecg', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(restecg = new_data)
new_data = ki.knn_impute(target=df['thalach'], attributes=df.drop(['thalach', 'binaryClass'], 1),
                                    aggregation_method="mean", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(thalach = new_data)
new_data = ki.knn_impute(target=df['exang'], attributes=df.drop(['exang', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(exang = new_data)
new_data = ki.knn_impute(target=df['slope'], attributes=df.drop(['slope', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(slope = new_data)
new_data = ki.knn_impute(target=df['ca'], attributes=df.drop(['ca', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.0)
df = df.assign(ca = new_data)
new_data = ki.knn_impute(target=df['thal'], attributes=df.drop(['age', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='jaccard', missing_neighbors_threshold=0.0)
df = df.assign(thal = new_data)
# print(df)
df.to_csv(path_or_buf='hasil_first.csv')