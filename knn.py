from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print('Train Dimension: ', X_train.shape)
print('Test Dimension: ', X_test.shape)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Target: ', y_test)
print('Predict: ', y_pred)
print('Accurate: ', model.score(X_test, y_test))


#find appropiate K value
accuracy = []
for K in range(3, 100):
    model = KNeighborsClassifier(n_neighbors = K)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
K = range(3, 100)
plt.plot(K, accuracy)
plt.show()