import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Load the training data from example.txt
X_train = np.genfromtxt('C:/Users/mayur/OneDrive/Desktop/CoursesSem2/ECC/project/KNN/example_dataset_knn.txt', delimiter=',', usecols=(0,1,2,3), dtype=float)
y_train = np.genfromtxt('C:/Users/mayur/OneDrive/Desktop/CoursesSem2/ECC/project/KNN/example_dataset_knn.txt', delimiter=',', usecols=(4), dtype=float)
print(X_train.shape)
print(y_train.shape)
# Load the test data from input_point.txt
X_test = np.genfromtxt('C:/Users/mayur/OneDrive/Desktop/CoursesSem2/ECC/project/KNN/input_points.txt', delimiter=',')
y_test = np.genfromtxt('C:/Users/mayur/OneDrive/Desktop/CoursesSem2/ECC/project/KNN/actual_labels.txt', delimiter=',')
# Create a KNN classifier object
k = 2
knn = KNeighborsClassifier(n_neighbors=k, p=2)

# Fit the classifier to the training data
knn.fit(X_train, y_train)
print(X_test.shape)
# Predict the label of the test data point
y_pred = knn.predict(X_test)
print(X_test.shape)

# Print the predicted label
print('Predicted label:', y_pred)
print(knn.score(X_test,y_test))
