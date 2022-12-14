from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('iris.data', header=None)
y = data.pop(4)
y = LabelEncoder().fit_transform(y)

km = DecisionTreeClassifier()
km.fit(data, y)
y_pred = km.predict(data)

with open('predict.txt', 'w') as f:
  print(*y_pred, sep='\n', end='', file=f)