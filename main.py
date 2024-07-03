# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from google.colab import drive

data = []
data_labels = []

def generate_datamatrix():
    for id in range(1, 41):
        for image_count in range(1, 11):
            image_path = f'/content/drive/MyDrive/archive/s{id}/{image_count}.pgm'
            image = Image.open(image_path)
            display(image)
            image_array = np.array(image)
            image_converted = image_array.flatten()  # get a vector of 10304 values
            data.append(image_converted)  # 400*10304
            data_labels.append(id)

generate_datamatrix()

data = np.array(data)
data_labels = np.array(data_labels)

train_indices = np.arange(0, data.shape[0], 2)
test_indices = np.arange(1, data.shape[0], 2)

train_data = data[train_indices]
train_data_labels = data_labels[train_indices]
test_data = data[test_indices]
test_data_labels = data_labels[test_indices]

def KNN(traindata, trainlabels, testdata, testlabels):
    n_neighbors = [1, 3, 5, 7]
    accuracy_list = []
    for n in n_neighbors:
        print(f"For n = {n} :")
        classifier = KNeighborsClassifier(n_neighbors=n, weights='distance')
        classifier.fit(traindata.T, trainlabels)
        prediction = classifier.predict(testdata.T)
        accuracy = accuracy_score(testlabels, prediction)
        accuracy_list.append(accuracy)
        print(f"Accuracy: {accuracy}")
        for j in range(len(prediction)):
            print(f"({j + 1}) Predicted: {prediction[j]} Actual label: {testlabels[j]}")
            if prediction[j] != testlabels[j]:
                print("Error")
    plt.plot(n_neighbors, accuracy_list)
    plt.title("Accuracy vs K neighbor value")
    plt.xlabel("K neighbor")
    plt.ylabel("Accuracy")
    plt.show()
    return accuracy_list

def PCA(train, test, train_labels, test_labels, alpha):
  training_mean = np.mean(train, axis=0)
  testing_mean = np.mean(test, axis=0)
  centered_train = train - training_mean
  centered_test = test - testing_mean
  covariance_matrix = np.cov(centered_train, bias = 1, rowvar = False)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
  idx = eigenvalues.argsort()[::-1]
  sorted_eigenvalues = eigenvalues[idx]
  sorted_eigenvectors = eigenvectors[:,idx]
  eigenvalues_sum = eigenvalues.sum()
  sum = 0
  fractional_variance = 0
  r = 0
  while (sum/eigenvalues_sum) < alpha:
    sum = sum + sorted_eigenvalues[r]
    r = r + 1
  reduced_matrix = sorted_eigenvectors[: , 0 : r]
  u_train = np.dot(reduced_matrix.T, centered_train.T)
  u_test = np.dot(reduced_matrix.T, centered_test.T)
  accuracy = KNN(u_train,train_labels,u_test,test_labels)
  print("projected matrix shape"+str(u_train.shape))
  return accuracy

print("PCA results")
k_values = np.array([1, 3, 5, 7])
alpha_values = [0.8, 0.85, 0.9, 0.95]
accuracies = []

for alpha in alpha_values:
    print(f"Alpha = {alpha}")
    acc = PCA(train_data, test_data, train_data_labels, test_data_labels, alpha)
    accuracies.append(acc)
accuracies_array = np.array(accuracies)
plt.figure()

for i, alpha in enumerate(alpha_values):
    plt.plot(k_values, accuracies_array[i], label=f"Alpha = {alpha}")

plt.title("Accuracy for each alpha Vs K value for PCA")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

def LDA(D, labels, k):
    D = np.asarray(D)
    labels = np.asarray(labels)
    n, d = D.shape
    classes = np.unique(labels)
    meanTotal = D.mean(axis=0)
    Sb = np.zeros((d, d), dtype=np.float32)
    Sw = np.zeros((d, d), dtype=np.float32)
    for i in classes:
        Di = D[labels == i]
        meani = Di.mean(axis=0)
        Sw += np.dot((Di - meani).T, (Di - meani))
        Sb += Di.shape[0] * np.outer(meani - meanTotal, meani - meanTotal)
    eigVals, eigVecs = np.linalg.eigh(np.dot(np.linalg.inv(Sw), Sb))
    idx = eigVals.argsort()[::-1][:k]
    eigenVecs = eigVecs[:, idx].real.astype(np.float32)
    return eigenVecs.T

def lda_classification(D_train, D_test, y_train, y_test, num_classes=40):
    k = num_classes - 1
    U = LDA(D_train, y_train, k)
    D_train_lda = D_train.dot(U.T)
    D_test_lda = D_test.dot(U.T)
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(D_train_lda, y_train)
    y_pred = classifier.predict(D_test_lda)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

lda_accuracy = lda_classification(train_data, test_data, train_data_labels, test_data_labels)
print("LDA Accuracy: ",lda_accuracy)

# New split: 7 instances for training and 3 for testing per subject
train_data73 = []
train_data_labels73 = []
test_data73 = []
test_data_labels73 = []
for i in range(40):
    subject_data = data[i*10:(i+1)*10]
    subject_labels = data_labels[i*10:(i+1)*10]
    train_data73.extend(subject_data[:7])
    train_data_labels73.extend(subject_labels[:7])
    test_data73.extend(subject_data[7:])
    test_data_labels73.extend(subject_labels[7:])

train_data73 = np.array(train_data73)
train_data_labels73 = np.array(train_data_labels73)
test_data73 = np.array(test_data73)
test_data_labels73 = np.array(test_data_labels73)

print("PCA results with new 7-3 split")
print("---------------------------")
alpha_values = [0.8, 0.85, 0.9, 0.95]
accuracies_pca = []

for alpha in alpha_values:
    print(f"Alpha = {alpha}")
    acc_pca = PCA(train_data, test_data, train_data_labels, test_data_labels, alpha)
    accuracies_pca.append(acc_pca)
k_values = np.array([1, 3, 5, 7])
plt.figure()
for i, alpha in enumerate(alpha_values):
    plt.plot(k_values, accuracies_pca[i], label=f"Alpha = {alpha}")
plt.title("Accuracy for each alpha Vs K value for PCA with new 7-3 split")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("LDA results with new 7-3 split")
print("---------------------------")
lda_accuracy_new_split = lda_classification(train_data73, test_data73, train_data_labels73, test_data_labels73)
print(f'LDA Accuracy with new 7-3 split: {lda_accuracy_new_split:.4f}')

data1 = []
data_label1 = []
face_nonface_test = []
face_nonface_test_label = []
face_nonface_training = []
face_nonface_training_label = []
def face_nonface_data(size):
    # Reading face images
    for id in range(1, 41):
        for image_count in range(1, 11):
            image_path = f'/content/drive/MyDrive/archive/s{id}/{image_count}.pgm'
            image = Image.open(image_path)
            image_array = np.array(image)
            image_converted = np.resize(image_array, (10304))
            data1.append(image_converted)
            data_label1.append("face")

    # Reading non-face images
    for i in range(size):
        if i < 10:
            image_path = f'/content/drive/MyDrive/car/car_000{i}.jpg'
        elif i < 100:
            image_path = f'/content/drive/MyDrive/car/car_00{i}.jpg'
        else:
            image_path = f'/content/drive/MyDrive/car/car_0{i}.jpg'

        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_resized = image.resize((92, 112))
        image_array = np.array(image_resized)
        image_converted = np.resize(image_resized, (10304))
        data1.append(image_converted)
        data_label1.append("non-face")

    # Splitting the mixed (face_nonface) dataset
    for i in range(len(data1)):
        if i % 2 == 0:
            face_nonface_test.append(data1[i])
            face_nonface_test_label.append(data_label1[i])
        else:
            face_nonface_training.append(data1[i])
            face_nonface_training_label.append(data_label1[i])

    return face_nonface_training, face_nonface_training_label, face_nonface_test, face_nonface_test_label

face_nonface_training, face_nonface_training_label, face_nonface_test, face_nonface_test_label = face_nonface_data(100)
dtrain=np.array(face_nonface_training)
dtest=np.array(face_nonface_test)
traininglabels=np.array(face_nonface_training_label)
testlabels=np.array(face_nonface_test_label)
ldaAccuracy100=lda_classification(dtrain,dtest, traininglabels, testlabels)
print(ldaAccuracy100)

face_nonface_training, face_nonface_training_label, face_nonface_test, face_nonface_test_label = face_nonface_data(300)
dtrain=np.array(face_nonface_training)
dtest=np.array(face_nonface_test)
traininglabels=np.array(face_nonface_training_label)
testlabels=np.array(face_nonface_test_label)
ldaAccuracy300=lda_classification(dtrain,dtest, traininglabels, testlabels)
print(ldaAccuracy300)

face_nonface_training, face_nonface_training_label, face_nonface_test, face_nonface_test_label = face_nonface_data(500)
dtrain=np.array(face_nonface_training)
dtest=np.array(face_nonface_test)
traininglabels=np.array(face_nonface_training_label)
testlabels=np.array(face_nonface_test_label)
ldaAccuracy500=lda_classification(dtrain,dtest, traininglabels, testlabels)
print(ldaAccuracy500)

accuracies= [ldaAccuracy100, ldaAccuracy300, ldaAccuracy500]
best_accuracies = [max(acc) if isinstance(acc, list) else acc for acc in accuracies]
sizes = [100, 300, 500]
plt.figure()
plt.plot(sizes, best_accuracies, marker='o')
plt.title("Dataset Size vs Accuracy")
plt.xlabel("Dataset Size")
plt.ylabel("Best Accuracy")
plt.grid(True)
plt.show()