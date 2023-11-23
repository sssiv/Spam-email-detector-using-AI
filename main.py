from data import Data, pd
from charts import Charts
from bayes import Bayes
from cnn import CNN
from svm import SVM

'''
It takes a while to load everything, so sometimes I 
want to just observe one dataset at a time
'''
def df_1():
    data = Data('data1.csv')
    df_data = data.df_dataset()
    return df_data
def df_2():
    data = Data('data2.csv')
    df_data = data.df_dataset()
    return df_data
def df_3():
    data = Data('data3.csv')
    df_data = data.df_dataset()
    return df_data

# Add datasets concatenated
def all_data():
    return pd.concat([df_1(), df_2(), df_3()])

# Get and clean datasets
df = all_data()

# Visual Data Display
'''
charts = Charts(df)
charts.wordplot()
charts.KDEplot()
charts.barchart()
charts.length_visual()
charts.word_frequency()
charts.scatterplot()
charts.plot_common_word_frequencies()
'''

# Example spam and ham emails to test models
ham = """
Hi Team,

Due to a scheduling conflict, I'm rescheduling this week's project meeting to Friday at 3 PM. Please update your calendars accordingly. 

We'll discuss the project's progress and next steps. If you have any specific items you'd like to add to the agenda, feel free to email me by Thursday noon.
Thanks for your understanding, and looking forward to our productive discussion.
Even if I try to add bias and put in words like 'viagra', these models will be able to pick properly.
Best,
Jordan
Project Manager
"""

spam = """
Hi mr or mrs usernameGenerator,
CLAIM NOW! CLAIM NOW! CLAIM NOW! CLAIM NOW! 
WEL HELLO THER, USER,
CONGRAUTUYGALASTIONS!!!!!!! CLAIM NOW! CLAIM NOW! 
WE GOT SELECTED INDIVUDUUALS HERE WHO GET $100,000 FOR FREE viagra, all you have to do is give us your credit!
Dont worry, all we are doing is simply funding you the given MONEY FOR FREE and maybe even even MONEY!
Just submit to us your credit NUMBER1 and NUMBER2 and NUMBER3 to us and we will getCreditInfo();
Make it MONEY FOR FREE AND you SPECIAL SELECTED for FREE viagra viagraFREE MONEY.
If you have Samsung Galaxy S24 Ultra, reply with Yes Or No for only $10! All viagra YOURS
Report user if not CLAIM NOW! viagra for as low as $1 , we just need you to download this link to do so. 
Come to us asap mr or mrs usernameGenerator 
"""
# Create CNN instance and assign loaded model and tokenizer
cnn = CNN(df)  
cnn.train_and_save_model('cnn_model.keras', 'tokenizer.pickle')
model, tokenizer = cnn.load_model_and_tokenizer('cnn_model.keras', 'tokenizer.pickle')

# Set model and tokenizer
cnn.model = model
cnn.tokenizer = tokenizer

prediction = cnn.preprocess_and_predict(spam)
print("Convolutional Neural-Network:\n\tGiven a spam Prediction:", prediction)
# print("This is a spam") if prediction > 0.5 else print("This is a NOT spam")
prediction = cnn.preprocess_and_predict(ham)
print("Convolutional Neural-Network:\n\tGiven a ham Prediction:", prediction)
print('-' * 20, '\n')

# SVM 
svm = SVM(df)
svm.train_and_save_model('svm_model.pkl', 'vectorizer.pickle')

svm_model, vectorizer = svm.load_model_and_vectorizer('svm_model.pkl', 'vectorizer.pickle')

email_vector = vectorizer.transform([spam])
prediction = svm_model.predict(email_vector)
print("Support Vector Machine:\n\tGiven a spam Prediction:", prediction)

email_vector = vectorizer.transform([ham])
prediction = svm_model.predict(email_vector)
print("Support Vector Machine:\n\tGiven a ham Prediction:", prediction)
print('-' * 20, '\n')