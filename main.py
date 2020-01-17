# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
import codecs
import sys
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from Database import Database
# from sklearn import tree
import numpy as np
import re
import nltk
from sklearn.datasets import load_files

# nltk.download('stopwords')
# nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords

import morfeusz2

morf = morfeusz2.Morfeusz()

db = Database('src/sejm_gov_pl_db.db')

df = db.read_as_pd("SELECT full_name, last_party, REPLACE(SUBSTR(speech_raw,INSTR(speech_raw,speech_title)), speech_title, '') speech_raw, date_ FROM speech_data sd JOIN portraits po ON sd.id_=po.id_")

documents = []
y = []
for sen in range(0, len(df['speech_raw'])):
    # Usunięcie nawiasów
    document = re.sub(r'\(.*?\)', ' ', str(df["speech_raw"][sen]))

    # Usunięcie znaków specjalnych
    document = re.sub(r'\W', ' ', str(document))

    # usunięcie pojedynczych znaków
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Usunięcie niepotrzebnych spacji
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # małe litery
    document = document.lower()

    # Lemantyzacja
    document = document.split()
    document = [morf.analyse(word)[-1][2][1] for word in document]
    document = ' '.join(document)


    # party
    # party = re.sub(r'\W', ' ', str(df["last_party"][sen]))
    ok = 0
    test = re.findall(r'.*platforma.*obywatelska.*', df["last_party"][sen], re.I)
    if test:
        df["last_party"][sen] = "PO"
        ok = 1
    test = re.findall(r'.*polskie.*stronni.*ludowe.*', df["last_party"][sen], re.I)
    if test:
        df["last_party"][sen] = "PSL"
        ok = 1
    test = re.findall(r'.*prawo.*i.*sprawiedliwość.*', df["last_party"][sen], re.I)
    if test:
        df["last_party"][sen] = "PiS"
        ok = 1
    test = re.findall(r'.*sojusz.*lewicy.*demokratycznej.*', df["last_party"][sen], re.I)
    if test:
        df["last_party"][sen] = "SLD"
        ok = 1

    if ok == 1:
        y.append(df["last_party"][sen])
        documents.append(document)

# print(documents)

# get stop words
file = codecs.open('src/polishST.txt', encoding='utf-8')
polishST = file.read().splitlines()
file.close()

vectorizer = CountVectorizer(max_features=1500, min_df=10, max_df=0.7, stop_words=polishST)
X = vectorizer.fit_transform(documents).toarray()
# print(X)

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
# print(X)

# Zestawy szkoleniowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model klasyfikacji tekstu szkoleniowego i przewidywanie nastrojów
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Ocena modelu
print("Ocena modelu: ")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# zamiast dwóch powyzek
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X = tfidfconverter.fit_transform(documents).toarray()

