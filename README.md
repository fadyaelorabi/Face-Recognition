# Face Recognition Project

This project focuses on performing face recognition using the ORL dataset, which contains images of 40 subjects. Each subject has 10 grayscale images of size 92x112 pixels. The goal is to identify the subject of a given image by leveraging machine learning techniques such as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Steps to Achieve the Goal](#steps-to-achieve-the-goal)
- [Results](#results)
- [Output Samples](#output-samples)

## Dataset

The ORL dataset can be found [here](https://www.kaggle.com/kasikrit/att-database-of-faces/). It contains 400 grayscale images (10 per subject) with dimensions of 92x112 pixels.

## Project Structure

The project follows these main steps:

1. **Download and Understand the Dataset**: Download the ORL dataset and understand its format.
2. **Generate Data Matrix and Label Vector**: Convert images to vectors and stack them to form a data matrix.
3. **Split Dataset for Training and Testing**: Split the dataset for training and testing.
4. **Classification Using PCA**: Perform classification using PCA and KNN.
5. **Classification Using LDA**: Perform classification using LDA and KNN.
6. **Classifier Tuning**: Tune the KNN classifier with different values of K.
7. **Bonus**: Additional tasks such as comparing face images with non-face images and experimenting with different training and testing splits.

## Steps to Achieve the Goal

### 1. Download the Dataset and Understand the Format

Download the dataset from the provided link and observe that it contains 40 subjects, each with 10 grayscale images of size 92x112 pixels.

### 2. Generate the Data Matrix and the Label Vector

- Convert each image into a vector of 10,304 values (92x112).
- Stack the 400 image vectors into a single data matrix \( D \) (400x10304).
- Generate a label vector \( y \) with integers from 1 to 40, corresponding to the subject IDs.

### 3. Split the Dataset for Training and Testing

- Use the odd rows of the data matrix \( D \) for training and even rows for testing.
- Split the label vector accordingly.

### 4. Classification using PCA

- Compute the projection matrix \( U \) using PCA.
- Define alpha as {0.8, 0.85, 0.9, 0.95}.
- Project the training and test sets separately using the same projection matrix.
- Use a simple nearest neighbor classifier to determine the class labels.
- Report accuracy for each alpha value.

### 5. Classification Using LDA

- Calculate the mean vector for each class.
- Compute the between-class scatter matrix \( S_b \) and within-class scatter matrix \( S_w \).
- Use 39 dominant eigenvectors for the projection matrix \( U \).
- Project the training and test sets separately.
- Use a simple nearest neighbor classifier.
- Report the accuracy for multiclass LDA and compare it with PCA results.

### 6. Classifier Tuning

- Set the number of neighbors in the K-NN classifier to 1, 3, 5, 7.
- Implement a preferred strategy for tie breaking.
- Plot or tabulate accuracy against the K value for PCA.

### 7. Bonus

- Compare face images with non-face images.
- Experiment with different training and test splits.

## Results

The project performs face recognition with both PCA and LDA methods, using K-NN as the classifier. The accuracy is measured for different values of alpha and K, and the results are plotted. Additionally, the impact of different training and testing splits and the inclusion of non-face images are explored.

## Output Samples

### PCA Results with both splits 50-50 , 70-30

**Trying each alpha with each k value:**

For alpha = 0.8:

- K = 1: Accuracy: 0.95
- K = 3: Accuracy: 0.93
- K = 5: Accuracy: 0.925
- K = 7: Accuracy: 0.91

For alpha = 0.85:

- K = 1: Accuracy: 0.95
- K = 3: Accuracy: 0.925
- K = 5: Accuracy: 0.915
- K = 7: Accuracy: 0.88

For alpha = 0.9:

- K = 1: Accuracy: 0.935
- K = 3: Accuracy: 0.93
- K = 5: Accuracy: 0.905
- K = 7: Accuracy: 0.87

For alpha = 0.95:

- K = 1: Accuracy: 0.94
- K = 3: Accuracy: 0.915
- K = 5: Accuracy: 0.895
- K = 7: Accuracy: 0.855

**Accuracy for Each Alpha vs K Value for PCA:**

![Accuracy for each Alpha vs K for PCA](images/alpha_vs_k_pca.png)

### LDA Results

**LDA Accuracy for different splits:**

- LDA Accuracy for 50-50 split: 0.94
- LDA Accuracy for 70-30 split: 0.95

### Bonus

**Face vs Non-Face Classification**

For dataset size 100:

- LDA Accuracy: 0.956

For dataset size 300:

- LDA Accuracy: 0.9788

For dataset size 500:

- LDA Accuracy: 0.982

**Dataset Size vs Accuracy for LDA**:
![Dataset Size vs Accuracy](images/dataset_size_vs_accuracy.png)
