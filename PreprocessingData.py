import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

StopWordSet = set(stopwords.words("english"))
not_stop = ["aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
            "mustn't", "needn't", "no", "nor", "not", "shan't", "shouldn't", "wasn't", "weren't", "wouldn't"]
for word in not_stop:
    StopWordSet.remove(word)

stemmer = nltk.PorterStemmer()


def tokenize(content):
    # transform from html to normal text
    review_text = BeautifulSoup(content, 'html.parser').get_text()
    # select the letter from the text
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # lower the letters
    lower_letters = letters_only.lower().split()
    # remove the stop words
    meaningful_words = [w for w in lower_letters if not w in StopWordSet]
    # Stemming
    Stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return ' '.join(Stemming_words)
    # return meaningful_words


train_path = r"D:\pycharm\BigdataProject\data\training.csv"
valid_path = r"D:\pycharm\BigdataProject\data\validation.csv"
test_path = r"D:\pycharm\BigdataProject\data\testing.csv"


def process_csv(data_path, csv_name):
    raw_data = pd.read_csv(data_path)

    raw_data.columns = ['Id', 'drugName', 'condition', 'review', 'date', 'usefulCount', 'sideEffect', 'rating']
    raw_data['clean_review'] = raw_data['review'].apply(tokenize)
    raw_data = raw_data.drop(['Id', 'drugName', 'condition', 'review', 'date', 'usefulCount', 'sideEffect'], axis=1)

    raw_data[['rating', 'clean_review']] = raw_data[['clean_review', 'rating']]
    raw_data.columns = ['review', 'rating']

    raw_data.to_csv(csv_name, index=None)

if __name__ == '__main__':
    # process_csv(train_path, "Mytraining.csv")
    # process_csv(valid_path, "Myvalidation.csv")
    process_csv(test_path, "Mytesting.csv")
