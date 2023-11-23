from encoder import *
# SVC Model
from sklearn.svm import SVC
from sklearn import metrics

# Makes Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SVM:
    # Initialize the class with a file path
    def __init__(self, df):
        # Load the data from the file into a DataFrame
        self.df = df

        # from encoder.py
        self.vectorizer, self.X, self.y = fit_transform_vectorizer(self.df)

        # Initialize two classifiers with specific hyperparameters
        self.classifiers = {
            "SVM": SVC(kernel='linear', probability=True)
        }

    # Function to train and evaluate the classifiers
    def train(self):
        # Split the data
        X_train, X_test, y_train, y_test = test_train_split_data(self.X, self.y)
        for name, clf in self.classifiers.items():
            # Train the classifier
            clf.fit(X_train, y_train)
            # Predict the labels for the test set
            y_pred = clf.predict(X_test)
            # Calculate accuracy
            accuracy = metrics.accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy*100:.2f}%")
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, name)

    # Function to plot the confusion matrix
    def plot_confusion_matrix(self, y_true, y_pred, classifier_name):
        matrix = metrics.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix for {classifier_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()