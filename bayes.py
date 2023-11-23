from encoder import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Makes Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Bayes:
    ## Initialize the class with a dataframe
    def __init__(self, df):
        # Load the dataframe
        self.df = df

        # from encoder.py
        self.vectorizer, self.X, self.y = fit_transform_vectorizer(self.df)

        # Initialize two classifiers with specific hyperparameters
        self.classifiers = {
            "Naive Bayes": MultinomialNB(alpha=True),
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

    # Function to predict the label for a given email content
    def predict_email(self, email_content):
        spam = False
        # Transform the email content using TF-IDF vectorizer
        email_vectorized = self.vectorizer.transform([email_content])
        for name, clf in self.classifiers.items():
            prediction = clf.predict(email_vectorized)
            confidence = clf.predict_proba(email_vectorized)
            if prediction == 0:
                spam = False
                print(f"{name} predicts: Ham with {confidence[0][0]*100:.2f}% confidence")
            else:
                spam = True
                print(f"{name} predicts: Spam with {confidence[0][1]*100:.2f}% confidence")
        return spam