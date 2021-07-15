from flask import Flask, render_template, request, url_for, redirect, render_template, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string as str
from string import punctuation
import string as str
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

df_true = pd.read_csv('True.csv')
df_false = pd.read_csv('Fake.csv')

df_false["Label"] = "false"
df_true["Label"] = "true"

false = df_false.drop(columns=['title','subject','date'])
true = df_true.drop(columns=['title','subject','date'])

merge = pd.read_csv('dataset_merge.csv')

x = merge['text']
y = merge['Label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

tfidf = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test.values.astype('U'))

loaded_model = pickle.load(open('model.pkl', 'rb'))

def remove_noise(text):
    for punctuation in str.punctuation:
        text = text.replace(punctuation, '')
    return text

listStopword = set(stopwords.words('english'))
def remove_stopword(text):
  words = text.split(" ")
  not_stopword = []
  for word in words:
    if word not in listStopword:
        not_stopword.append(word)
  text = ' '.join(not_stopword)
  return text

false['Text'] = (false['text'].str.lower()).apply(remove_stopword,remove_noise)

true['Text'] = (true['text'].str.lower()).apply(remove_stopword,remove_noise)

vectorizer = TfidfVectorizer(min_df=0.35, max_df=0.80)

tfIdf = vectorizer.fit_transform(false['Text'])
tfidf_false = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
tfidf_false = tfidf_false.sort_values('TF-IDF', ascending=False)
print (tfidf_false.head(25))

tfIdf = vectorizer.fit_transform(true['Text'])
tfidf_true = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
tfidf_true = tfidf_true.sort_values('TF-IDF', ascending=False)
print (tfidf_true.head(25))

tfidf_false = tfidf_false.rename_axis('word').reset_index()

tfidf_true = tfidf_true.rename_axis('word').reset_index()

tfidf_false.to_csv('word false.csv')

tfidf_true.to_csv('word true.csv')

app = Flask(__name__)

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
   print(input_data)
   status = loaded_model.predict(tfidf.transform(input_data))
   print(status)

  #  falseWordProofing = {} 
  #  trueWordProofing = {}
  #  if status == 'false':
  #    falseWordProofing = falsewordproofing(news2)
  #  else :
  #     trueWordProofing = trueWordProofing(news2)

   if status == 'false' :
    #  return jsonify({"hasil":0, "wordProofing":falsewordProofing})
      return 'Hoax'
   else :
    #  return jsonify({"hasil":1, "WordProofing":trueWordProofing})
      return 'Reliable'
   
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