from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('iris.data', header=None)
data.pop(4)

km = KMeans(n_clusters=3)
km.fit(data)
y_pred = km.predict(data)

with open('predict.txt', 'w') as f:
  print(*y_pred, sep='\n', end='', file=f)