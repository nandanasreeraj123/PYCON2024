from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle


data = load_iris()
print(len(data))

model=KNeighborsClassifier(n_neighbors=3)
model.fit(data.data,data.target)

with open('iris_model.pkl','wb') as f:
    pickle.dump(model,f)

