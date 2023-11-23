# Visual libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Dataframe
import pandas as pd

# String libraries
import nltk
from nltk.corpus import stopwords

# Used to find any type of most common type of whatever data you pick
from collections import Counter

from encoder import fit_vectorizer

class Charts:
    def __init__(self, df):
        self.df = df
        # Split the dataset into spam and ham
        self.spam_texts = self.df[self.df['label_num'] == 1]['text']
        self.ham_texts = self.df[self.df['label_num'] == 0]['text']

        # Use Tfidf Vectorizer to get word frequencies
        self.vectorizer = fit_vectorizer(df)
        self.spam_word_counts = self.vectorizer.transform(self.spam_texts).sum(axis=0)
        self.ham_word_counts = self.vectorizer.transform(self.ham_texts).sum(axis=0)    #fit or fit_transform?

        # Create word lists and their counts
        self.words = self.vectorizer.get_feature_names_out()
        self.spam_counts = self.spam_word_counts.tolist()[0]
        self.ham_counts = self.ham_word_counts.tolist()[0]

        # Filters out non-english words
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        # Get the most common words for spam and ham
        self.df['tokens'] = self.df['text'].apply(nltk.word_tokenize)
        self.df['tokens'] = self.df['tokens'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word.lower() not in self.stop_words])
        self.spam_words = [word for sublist in self.df[self.df['label_num'] == 1]['tokens'].tolist() for word in sublist]
        self.ham_words = [word for sublist in self.df[self.df['label_num'] == 0]['tokens'].tolist() for word in sublist]
    
    
    # List of the most common words found in all emails
    def wordplot(self):
        self.text_corpus = " ".join(self.df['text'])
        self.wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(self.text_corpus)
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    
    '''
    Counts the number of words in each email and plots them into a smoothened plot
    almost like taking a scatterplot and organizing it so that when you
    draw a line through it, it looks smooth
    '''
    def KDEplot(self):
        # Create a KDE plot
        plt.figure(figsize=(10, 8))
        sns.kdeplot(self.spam_counts, label='Spam', color='purple')
        sns.kdeplot(self.ham_counts, label='Ham', color='green')
        plt.xlabel('Word Counts')
        plt.ylabel('Density')
        plt.title('KDE Plot of Word Frequencies in Spam and Ham Texts')
        plt.legend()
        plt.show()

    # Visual aid to make sure the data is balanced
    def barchart(self):
        count_Class=pd.value_counts(self.df['label'], sort=True)
        count_Class.plot(kind='bar', color= ["blue", "orange"])
        plt.title('Bar chart')
        plt.show()
    
    # Shows how many messages have how many characters in them
    def length_visual(self):
        # Plot for distribution lenth of text
        self.df['length'] = self.df['text'].apply(len)
        self.df[self.df['length']==self.df['length'].max()]['text']
        self.df.hist(column='length',by='label',figsize=(12,8))
        plt.title('Spam')
        plt.show()
    
    # Shows how moften there are repeating words
    def word_frequency(self):
        # Create a histogram
        plt.figure(figsize=(10, 8))
        plt.hist(self.spam_counts, bins=30, alpha=0.5, color='purple', label='Spam')
        plt.hist(self.ham_counts, bins=30, alpha=0.5, color='green', label='Ham')
        plt.xlabel('Word Counts')
        plt.ylabel('Frequency')
        plt.title('Histogram of Word Frequencies in Spam and Ham Texts')
        plt.legend()
        plt.show()
    
    # Takes the most common words in both ham and spam then compares them
    # Sometimes words will have a frequency of 0 in one and a lot in the other
    # This is good because it helps us tell which words are in ham/spam and not the other
    def scatterplot(self):
        spam_word_freq = Counter(self.spam_words)
        ham_word_freq = Counter(self.ham_words)

        # Select the top 30 words
        top_spam_words = [item[0] for item in spam_word_freq.most_common(30)]
        top_ham_words = [item[0] for item in ham_word_freq.most_common(30)]

        # Create the scatter plot
        plt.figure(figsize=(15, 8))

        # Plot the word frequencies for ham and spam
        for word in set(top_spam_words + top_ham_words):
            plt.scatter(word, spam_word_freq[word], color='red', label='Spam', marker='x')
            plt.scatter(word, ham_word_freq[word], color='blue', label='Ham', marker='o')

        # Add title, labels, and legend
        plt.title('Top 30 Word Frequencies for Spam and Ham Emails within range of 700')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()

    # Shows most common words in both ham and spam (in other words, noise)        
    def plot_common_word_frequencies(self):
        # Split the datasets
        spam_texts = self.df[self.df['label'] == 'spam']['text'].tolist()
        ham_texts = self.df[self.df['label'] == 'ham']['text'].tolist()

        # Get word counts
        spam_words = Counter(" ".join(spam_texts).split())
        ham_words = Counter(" ".join(ham_texts).split())

        # Get the most common words in spam and ham
        common_words = set(spam_words).intersection(set(ham_words))
        most_common = common_words # Here you could filter for the top N words

        # Get frequencies for the plot
        word_freq = [(word, spam_words[word], ham_words[word]) for word in most_common]
        word_freq.sort(key=lambda x: x[1] + x[2], reverse=True)  # Sort by combined frequency

        # Take only the top N words for cleaner plotting
        top_n = 50
        word_freq = word_freq[:top_n]

        # Create the scatterplot
        plt.figure(figsize=(15, 10))
        for word, spam_f, ham_f in word_freq:
            plt.scatter(spam_f, ham_f, alpha=0.5)
            plt.text(spam_f, ham_f, word)

        # Plotting details
        plt.xlabel('Spam Frequency')
        plt.ylabel('Ham Frequency')
        plt.title('Word Frequencies in Spam vs. Ham')
        plt.xscale('log')  # Using log scale if needed
        plt.yscale('log')  # Using log scale if needed
        plt.show()