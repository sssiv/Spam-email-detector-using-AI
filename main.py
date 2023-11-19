from data import *
from charts import *
from model import *

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

# Visual Data Display
charts = Charts(df)
charts.wordplot()
charts.KDEplot()
charts.barchart()
charts.length_visual()
charts.word_frequency()
charts.scatterplot()
charts.plot_common_word_frequencies()

# Testing each Model
# CNN
cnn = CNN(df)
cnn_model = cnn.train()

'''
text, label = df['text'].sample().iloc[0], df['label_num'].sample().iloc[0]
email_content = input(text)
def test(model):
    if model.predict_email(email_content) == True and label == 1:
        print("E-mail identified as spam guessed correctly!\n")
    elif model.predict_email(email_content) == False and label == 0:
        print("E-mail identified as ham guessed correctly!\n")
    else:
        print("E-mail identified incorrectly\n")
'''
# Naive Bayes
bayes_model = Bayes(df)
bayes_model.train()
#test(bayes_model)

# SVM 
svm_model = SVM(df)
svm_model.train()
#test(svm_model)