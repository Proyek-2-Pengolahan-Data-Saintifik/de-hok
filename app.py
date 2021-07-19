from flask import Flask, render_template, request, url_for, redirect, render_template
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string as str
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

# Define the TFIDF with max value 0.7
tfidf = TfidfVectorizer(max_df=0.7)

# Loaded model and words for word proofing
loaded_model = pickle.load(open('model.pkl', 'rb'))
tfidf_false = pd.read_csv('word_false.csv')
tfidf_true = pd.read_csv('word_true.csv')

# Define list of stopword to preprocessing input
listStopword = set(stopwords.words('english'))

def remove_noise(text):
    for punctuation in str.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_stopword(text):
  words = text.split(" ")
  not_stopword = []
  for word in words:
    if word not in listStopword:
        not_stopword.append(word)
  text = ' '.join(not_stopword)
  return text

def falseWordProofing(inputData):
  words = inputData.split(" ")
  false_word = []
  for Word in words:
    if tfidf_false['word'].str.contains(Word).any():
        false_word.append(Word)
  result = []
  [result.append(x) for x in false_word if x not in result]
  if '' in result: result.remove('')
  return {'resultFalse':result}

def trueWordProofing(inputData):
  words = inputData.split(" ")
  true_word = []
  for Word in words:
    if tfidf_true['word'].str.contains(Word).any():
        true_word.append(Word)
  result = []
  [result.append(x) for x in true_word if x not in result]
  if '' in result: result.remove('')
  return {'resultTrue':result}

def preprocess_input(text):
  return remove_stopword(remove_noise(text.lower()))

def fake_news_det(news):
  news = news.lower()
  news1 = remove_stopword(news)
  news2 = remove_noise(news1)
  input_data = [news2]
  status = loaded_model.predict(tfidf.transform(input_data))

  return ('Hoax') if (status == 'false') else ('Realiable')
   
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/aboutUs', methods=['GET', 'POST'])
def aboutUs():
    if request.method == 'POST' :
        return redirect(url_for('index'))
    return render_template('aboutUs.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        result = fake_news_det(data['news'])
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)