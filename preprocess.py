import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(data_set):

    with open(data_set, 'r', encoding='utf-8') as file:
        text = file.read()

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)

def save_data_to_file(preprocessed_text, training_data):
    # Open the file in write mode and save the preprocessed text
    with open(training_data, 'w', encoding='utf-8') as file:
        file.write(preprocessed_text)

data_set = r"C:\Users\abith\Downloads\Dataset.txt"
preprocessed_text = preprocess_text(data_set)