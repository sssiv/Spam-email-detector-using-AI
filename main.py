from data import *
from charts import *
import pickle
from bayes import Bayes
from cnn import CNN
from svm import SVM
from keras.models import load_model

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

def all_data():
    return pd.concat([df_1(), df_2(), df_3()])

# Get and clean datasets
df = all_data()

cnn = CNN(df)  # 'df' can be any DataFrame, used for initializing CNN
# Create CNN instance and assign loaded model and tokenizer
# Assuming you have a DataFrame 'df' with your training data
cnn.train_and_save_model('cnn_model.h5', 'tokenizer.pickle')

def load_model_and_tokenizer(model_filename, tokenizer_filename):
    # Load the tokenizer reading as binary
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Load and return the model and tokenizer file
    return load_model(model_filename), tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer('cnn_model.h5', 'tokenizer.pickle')

cnn.model = model
cnn.tokenizer = tokenizer
# Use the model for prediction
ham = """
Hi Team,

Due to a scheduling conflict, I'm rescheduling this week's project meeting to Friday at 3 PM. Please update your calendars accordingly. 

We'll discuss the project's progress and next steps. If you have any specific items you'd like to add to the agenda, feel free to email me by Thursday noon.

Thanks for your understanding, and looking forward to our productive discussion.

Best,
Jordan
Project Manager
"""

spam = """
Hi mr or mrs usernameGenerator,
	
A user just logged into your Facebook account from a new device: ( Samsung Galaxy S24 Ultra ) Location: Dmitrov, Oblast de Moscou, Russie, 141101.
We are sending you this email to verify it's really you.
If you have Samsung Galaxy S24 Ultra, reply with Yes Or No
Report user if not recognized , we just need you to download this link to do so. 
Thanks,
The Facebook Team 
"""

prediction = cnn.preprocess_and_predict(ham)
print("Given a ham Prediction:", prediction)

prediction = cnn.preprocess_and_predict(spam)
print("Given a spam Prediction:", prediction)
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

# Naive Bayes
bayes_model = Bayes(df)
bayes_model.train()
#test(bayes_model)
'''

# SVM 
svm_model = SVM(df)
svm_model.train()
#test(svm_model)
