# For dataframe
import pandas as pd

# Natural Language Toolkit. Has english and tokenizing tools
import nltk
from nltk.corpus import words, stopwords
from nltk.tokenize import word_tokenize

# Regular Expression: string ideitifying and manipulating libraries 
import re

# Used to Normalize data
from sklearn.preprocessing import MinMaxScaler


# Takes in the csv file
# Removes junk data
# Describes data
# Removes null values
class Data():
    def __init__(self, filename):
        # Download the necessary resources
        nltk.download('words')
        nltk.download('stopwords')
        nltk.download('punkt')

        self.df = pd.read_csv(filename, low_memory=False)
        #self.describe_data()
        self.null_handler()
        self.remove_outliers()
        self.normalize_data()
        self.balance_data()
        self.remove_noise()

    print('\n\n')
    print('-' * 172)

    def keep_real_words(self, email_content):
        # Set of English words
        self.english_words = set(words.words())
        # List of words to exclude
        exclude_words = ["enron", "\\", "text_", "text_r",
                        "\r\n", "hpl actuals", "hpl", "ect", "hou",
                        "subj", "cc", "mmbtu", "sld", "th",
                        "e", "anks", "wi"]

        # Explicitly remove occurrences of "Subject" using regex
        email_content = re.sub(r"\bSubject\b", '', email_content, flags=re.IGNORECASE)

        # Tokenize the email text
        tokens = word_tokenize(email_content)

        # Filter the tokens
        real_words = [
            token for token in tokens
            if token.isalpha()   # Check if token consists of only alphabets
            and len(token) > 1  # Consider only tokens that have a length greater than 1
            and token.lower() not in exclude_words  # Exclude unwanted words
            and token.lower() in self.english_words  # Check for presence in English word dataset
        ]

        # Reconstruct the cleaned text
        cleaned_text = ' '.join(real_words)

        return cleaned_text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        # Reconstruct the text from tokens
        clean_text = ' '.join(filtered_tokens)
        return clean_text
    
    def describe_data(self):
        # Data Description
        print('\n\n')
        print("Describe Data")
        print(self.df.describe)
        print('\n\n')

        # Head of dataset
        print('\n\n')
        print("Data Sample from Head")
        print(self.df.head())
        print('\n\n')

        # datatype and null info
        print('\n\n')
        print("Data Info")
        print(self.df.info())
        print('\n\n')

        # Averages
        print('\n\n')
        print("Averages")
        print(self.df.describe())
        print('\n\n')

    # Define a function that handles the conditional logic
    def fill_missing_labels(self, row):
        if pd.isna(row['label']) and not pd.isna(row['label_num']):
            return "spam" if row['label_num'] == 1 else "ham"
        if pd.isna(row['label_num']) and not pd.isna(row['label']):
            return 1 if row['label'].lower() == "spam" else 0
        return row['label']

    def null_handler(self):
        # How many data enteries are null
        print('\n\n')
        print("Null Count")
        print(self.df.isnull().sum())
        print('\n\n')

        print('\n\n')
        print("Filling missing data for label and label_num")
        self.df['label'] = self.df.apply(self.fill_missing_labels, axis=1)
        # This will fill the 'label_num' based on 'label' as the conditions specify
        self.df['label_num'] = self.df.apply(lambda row: 1 if row['label'] == "spam" else 0, axis=1)
        print(self.df.isnull().sum())
        print('\n\n')

        # Drop null data
        print('\n\n')
        print("Dropping null data")
        # Removes null data
        # Drop rows where both 'label' and 'label_num' are missing, or 'text' is missing
        self.df.dropna(subset=['label', 'label_num', 'text'], how='any', inplace=True)
        # Confirm null data has been dropped
        print(self.df.isnull().sum(), '\n')
        print(self.df.info(), '\n')
        print('\n\n')

    def normalize_data(self):
        # Create a MinMaxScaler
        scaler = MinMaxScaler()

        # Identify numeric columns
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Normalize only the numeric columns
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
 
    def balance_data(self):
        ham_df = self.df[self.df['label'] == 'ham']
        spam_df = self.df[self.df['label'] == 'spam']

        # Calculate the size difference
        size_difference = abs(len(ham_df) - len(spam_df))

        # Determine which category has more entries
        if len(ham_df) > len(spam_df):
            # Sample the necessary number of ham instances to remove
            ham_to_remove = ham_df.sample(n=size_difference, random_state=1)  # n for absolute number
            # Remove the sampled instances
            reduced_ham_df = ham_df.drop(ham_to_remove.index)
            balanced_df = pd.concat([reduced_ham_df, spam_df], axis=0)
        elif len(spam_df) > len(ham_df):
            # Sample the necessary number of spam instances to remove
            spam_to_remove = spam_df.sample(n=size_difference, random_state=1)  # n for absolute number
            # Remove the sampled instances
            reduced_spam_df = spam_df.drop(spam_to_remove.index)
            balanced_df = pd.concat([ham_df, reduced_spam_df], axis=0)
        else:
            # If they are already balanced, no need to sample or remove
            balanced_df = self.df

        # Shuffle the balanced DataFrame
        '''
        frac:
            How much (%) of the rows to return, 100%
        random_state:
            the seed for the rng of the sampler
        reset_index(drop=True):
            Each row has an index that shows what order it was in before shuffling. 
            When you shuffle the DataFrame, you mix up all the rows. 
            The .reset_index(drop=True) part is like giving each row a new number to match the new order. 
            And drop=True means you're erasing the old numbers so you don't get confused about what order the rows are now in.
        '''
        balanced_df = balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)

        # Now 'balanced_df' has an equal number of 'ham' and 'spam' data
        self.df = balanced_df
    # detect and remove outliers
    def remove_outliers(self):
        '''
        String Check: 
            Make sure to keep only those rows where the 'text' column 
            is of type string and get rid of any non-copatiable types
        '''
        self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))]
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Text Lengths: 
            We calculate and measure how long each 'text' 
            is and prepare to split in quartiles
        '''
        text_lengths = self.df['text'].apply(len)
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Quartiles: 
            Comparing text by length,
            Q1 is the length of the text at the 25% mark,
            Q3 is at the 75% mark.
            The distance between Q1 and Q3 is the IQR (Interquartile Range).
        '''
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        Q1 = text_lengths.quantile(0.25)
        Q3 = text_lengths.quantile(0.75)
        IQR = Q3 - Q1
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Outlier Bounds: 
            Using the results from Q1, Q3, and IQR),
            we decide two height thresholds:
            Very Short texts (lower_bound) and
            Very Long Texts (upper_bound).
            These legnths are considered unusual and outliers.
        '''
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Filtering: 
            We keep only the 'text' entries that have lengths between our thresholds.
            Those outside this range are considered the outliers
            They are now removed
        '''
        self.df = self.df[(text_lengths >= lower_bound) & (text_lengths <= upper_bound)]
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Removing Redunancy:
            There's a lot of useless repeated words that have
            no real meaning, so we use nltk libraries to 
            remove them and get better results
        '''
        self.df['text'] = self.df['text'].apply(self.keep_real_words)
        self.df['text'] = self.df['text'].apply(self.remove_stopwords)
        # Removes words that have too large of a difference in frequency
        self.remove_word_class_bias()
        
    def remove_word_class_bias(self):
        self.ham_texts = self.df[self.df['label'] == 'ham']['text']
        self.spam_texts = self.df[self.df['label'] == 'spam']['text']

        # Tokenize the text and count the frequency of each word for both classes
        self.ham_word_freq = nltk.FreqDist(' '.join(self.ham_texts).split())
        self.spam_word_freq = nltk.FreqDist(' '.join(self.spam_texts).split())

        # Compute the difference in frequency for each word between the two classes
        outlier_words = []
        junk_words = ["c", "ce", "st", "n", "e", "r", "se", "ami"]

        # Ensure 'text' column contains strings
        self.df['text'] = self.df['text'].astype(str)

        # Remove these stubborn words that just wont get out for some reason
        self.df['text'] = self.df['text'].str.replace(r'\br\b', '', regex=True)  # for word 'r'
        self.df['text'] = self.df['text'].str.replace(r'\bre\b', '', regex=True)  # for word 're'
        self.df['text'] = self.df['text'].str.replace(r'\bse\b', '', regex=True)  # for word 'se'
        self.df['text'] = self.df['text'].str.replace(r'\bami\b', '', regex=True)  # for word 'ami'

        for word, ham_freq in self.ham_word_freq.items():
            if word in junk_words:
                continue
            if len(word) <= 2:
                continue
            if not word.isalpha():
                continue
            spam_freq = self.spam_word_freq.get(word, 0)
            if abs(ham_freq - spam_freq) > 750:  # Threshold value
                outlier_words.append(word)

        # Remove outlier words from the dataset
        for word in outlier_words:
            self.df['text'] = self.df['text'].str.replace(rf'\b{word}\b', ' ', regex=True)

    def remove_noise(self):
        # Find common words between the two distributions
        common_words = set(self.ham_word_freq).intersection(set(self.spam_word_freq))

        # Compute the total frequency of each common word
        common_word_freq = {word: self.ham_word_freq[word] + self.spam_word_freq[word] for word in common_words}

        # Sort the common words by their total frequency (descending)
        sorted_common_words = sorted(common_word_freq.items(), key=lambda item: item[1], reverse=True)

        # Choose a threshold or number of top common words to remove
        num_words_to_remove = 250  # for example, take the top 100 most common words
        words_to_remove = set(word for word, freq in sorted_common_words[:num_words_to_remove])

        # Remove these common words from the dataset
        for word in words_to_remove:
            self.df['text'] = self.df['text'].str.replace(rf'\b{word}\b', ' ', regex=True)   
       
    def df_dataset(self):
        return self.df