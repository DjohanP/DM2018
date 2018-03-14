import knn_impute as ki
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# imputer
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
del df['ca']
new_data = ki.knn_impute(target=df['thal'], attributes=df.drop(['thal', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(thal = new_data)
new_data = ki.knn_impute(target=df['thal'], attributes=df.drop(['thal', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=100, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(thal = new_data)
new_data = ki.knn_impute(target=df['thal'], attributes=df.drop(['thal', 'binaryClass'], 1),
                                    aggregation_method="mode", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(thal = new_data)
# print(df)
df.to_csv(path_or_buf='hasil_first.csv')
df2 = pd.read_csv('hasil_first.csv',index_col=0)
df=df.replace('male', 1)
df=df.replace('female',0)
df=df.replace('f',0)
df=df.replace('t',1)
df=df.replace('no',0)
df=df.replace('yes',1)
df=df.replace('P',1)
df=df.replace('N',0)

data_thal = list(df['thal'])
chest_pain = list(df['chest_pain'])
restecg=list(df['restecg'])
slope=list(df['slope'])

lb = preprocessing.LabelBinarizer()
lb.fit(data_thal)
data_thal_binarized = lb.transform(data_thal)

lb2 = preprocessing.LabelBinarizer()
lb2.fit(chest_pain)
data_chest_binarized = lb2.transform(chest_pain)

lb3 = preprocessing.LabelBinarizer()
lb3.fit(restecg)
data_restecg_binarized = lb3.transform(restecg)

lb4 = preprocessing.LabelBinarizer()
lb4.fit(slope)
data_slope_binarized = lb3.transform(slope)

for i in range(0, 3):
    df['thal-'+str(i)] = data_thal_binarized[:, i]
for i in range(0, 4):
    df['chest-'+str(i)] = data_chest_binarized[:, i]
for i in range(0, 3):
    df['restecg-'+str(i)] = data_restecg_binarized[:, i]
for i in range(0, 3):
    df['slope-'+str(i)] = data_slope_binarized[:, i]

del df['thal']
del df['chest_pain']
del df['slope']
del df['restecg']

pca = PCA(n_components=4)
pca.fit(df)
data2=pca.transform(df)
y=df['binaryClass']
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size = 0.2, random_state = 0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
benar=(y_test == y_pred).sum()
print benar/59.0*100

pca = PCA(n_components=7)
pca.fit(df)
data2=pca.transform(df)
y=df['binaryClass']
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size = 0.2, random_state = 0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
benar=(y_test == y_pred).sum()
print benar/59.0*100

pca = PCA(n_components=11)
pca.fit(df)
data2=pca.transform(df)
y=df['binaryClass']
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size = 0.2, random_state = 0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
benar=(y_test == y_pred).sum()
print benar/59.0*100

dataset = df.values
kmeans = KMeans(n_clusters=nclust,init='k-means++').fit(dataset)
labels = kmeans.labels_
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels)

datasetBaru = dataset
for i in range(0, nclust):
    count = np.count_nonzero(labels == i)
    if count <= 3:
        indexDelete = np.where(labels == i)
        dataset = np.delete(dataset, indexDelete, axis=0)
        labels = np.delete(labels, indexDelete, axis=0)

kmeans_after = KMeans(n_clusters=3).fit(dataset)
labels_after = kmeans_after.labels_
        
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels_after)
plt.show()