from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
from encoder import test_train_split_data

'''
    RESULTS EXPLINATION
    Epoch 1/5
    46/46 [==============================] - 1s 12ms/step - loss: 0.6524 - accuracy: 0.6681 - val_loss: 0.5863 - val_accuracy: 0.6842

    Epoch:
        Unix measurment of time

    46/46: 
        The robot looked at 46 sets of apples and oranges.
        loss: 0.6524: After the first lesson, the robot made some mistakes.
        "0.6524" is a way to measure those mistakes. The smaller this number, the fewer mistakes the robot made.

    accuracy: 0.6681: 
        This tells us the robot correctly identified apples and oranges 66.81% of the time during its training.

    val_loss: 0.5863: 
        Now, we showed the robot new sets of apples and oranges it has never seen before. It made mistakes, and this "0.5863" measures those.

    val_accuracy: 
        0.6842: On these new sets, the robot was correct 68.42% of the time.
'''
class CNN:
    # Initialize the class with a dataframe
    def __init__(self, df):
        # Load the dataframe
        self.df = df

        # Tokenize the text
        '''
        Tokenizing:
            We keep the 5000 most common words
            Collect all the words
            Convert words into numerical sequences
            Make sure all sequences are the same maximum length
        '''
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(self.df['text'])
        self.X = self.tokenizer.texts_to_sequences(self.df['text'])
        self.X = pad_sequences(self.X, padding='post', maxlen=100)  # Assuming each text sequence is truncated/padded to a max length of 100
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Encoding:
            Converts categorical labels into a numerical format
            Using the encoder, fit and transform the labels 
            which assignes unique values to each label
        '''
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df['label'])

    def build_model(self):
        '''
        Sequential():
            Think of this like building a LEGO tower.
            Each piece you add goes on top of the other, one by one.
            It's the base where you'll start stacking your layers.
        '''
        model = Sequential()
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Embedding Layer:
            Imagine you have a box of colored pencils, each with a different color 
            representing a word in your emails. But there are too many 
            colored pencils (words) - pretend 5000, You want a simpler way to 
            represent these colors when drawing (processing data).
            So, you decide that you'll use a smaller set of just 
            128 special pencils that can somehow capture the essence of 
            those 5000 colors. The Embedding layer helps in making this magic happen.
        '''
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Conv1D Layer:
            Think of a magnifying glass used to spot patterns in a drawing
            Conv1D does that for sequences of words. 
            With "128" different magnifying glasses each focusing on 
            a specific pattern, and "10" being the size of the patterns they're trying to find.
            It's looking for special combinations of words or characters.
        '''
        model.add(Conv1D(500, 10, activation='swish'))   # was relu
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        GlobalMaxPooling1D Layer:
            Imagine you drew lots of pictures and you want to pick the 
            brightest color from each drawing. GlobalMaxPooling takes the most 
            standout feature (brightest color in our analogy) from the Conv1D results.
            It's like extracting the highlight from each set of patterns.
        '''
        model.add(GlobalMaxPooling1D())
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Dense Layer:
            Picture this as a team of detectives in a room discussing clues.
            They are all connected and share their findings.
            The "swish" activation is like their secret language, 
            helping them piece together complex mysteries.
            This layer makes major decisions based on the previous layers.
        '''
        model.add(Dense(10, activation='swish'))    # was relu
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Dropout Layer:
            Think of playing a memory game where occasionally, you remove 
            a few cards to make it more challenging. Dropouts do that by 
            randomly turning off some connections, making sure that the network 
            doesn't cheat by relying too much on one particular clue.
            In this case is drops out 50% of the data
        '''
        model.add(Dropout(0.5))
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Output Dense Layer:
            Imagine you're at a game show where,
            after going through multiple rounds and challenges,
            you have to press one of two big buttons at the end: "Yes" or "No".
            The final decision depends on all the previous rounds.
            In our model, the Dense(1, activation='sigmoid') layer is like this final round of the game show.
            The "1" means we have only one button or decision to make at the end,
            which will tell us whether an email is "Spam" (Yes) or "Not Spam" (No).
        '''
        model.add(Dense(1, activation='sigmoid'))   # binary classification
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        '''
        Compile: prepares the model for training

        Optimizer (optimizer='adam'):
            This is the strategy or method your model will use to adjust its weights based on the data.
            Think of it like adjusting knobs to get the best output.
            'Adam' is a popular choice because it's efficient and requires little memory.
            It's like choosing a reliable tool to help build something.

        Loss Function (loss='binary_crossentropy'):
            This is how your model measures how right or wrong its predictions are.
            'Binary_crossentropy' is used because you're doing binary classification (like sorting things into two boxes: spam or not spam).
            The model wants to be as less "wrong" as possible,
            so this function helps it understand the difference between its predictions and the actual results.
        '''
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #-----------------------------------------------------------------------#
        #-----------------------------------------------------------------------#
        return model

    def train(self):
        self.model = self.build_model()
        X_train, X_test, y_train, y_test = test_train_split_data(self.X, self.y)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=10)
        return self.model
    
    def preprocess_and_predict(self, email):
        # Preprocess the email
        sequence = self.tokenizer.texts_to_sequences([email])
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)

        # Predict
        prediction = self.model.predict(padded_sequence)
        return prediction
    
    def train_and_save_model(self, model_filename, tokenizer_filename):
        # make object instance 
        cnn = CNN(self.df)

        # Train model
        self.model = cnn.train()

        # Save model after training
        self.model.save(model_filename)
        
        # Write binary
        with open(tokenizer_filename, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_model_and_tokenizer(model_filename, tokenizer_filename):
        # Load in the model
        cnn_model = load_model(model_filename)

        # Load the tokenizer reading as binary
        with open(tokenizer_filename, 'rb') as handle:
            tokenizer = pickle.load(handle)

        return cnn_model, tokenizer