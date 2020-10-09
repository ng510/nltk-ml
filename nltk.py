# Datenquelle: https://www.kaggle.com/snap/amazon-fine-food-reviews
# 
# Davon betrachten wir nur die ersten 10.000 Zeilen.
#
# Model lernt zwischen "guten" und "schlechten" adjektiven zu unterscheiden mit Hilfe von Amazon Rezensionen

import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("./Reviews_10000.csv.bz2")
print(df.head())

texts = df["Text"]

texts_transformed = []
# Splitte die reviews in Sätze bzw. Wörter
# tagge anschließend um welche Art es sich handelt (Nomen, Adjektiv, etc.)
for review in texts: 
    sentences = nltk.sent_tokenize(review)
    adjectives = []
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words_tagged = nltk.pos_tag(words)
        
        for word_tagged in words_tagged:
            if word_tagged[1] == "JJ": #gebe nur Adjektive
                adjectives.append(word_tagged[0])
                
    texts_transformed.append(" ".join(adjectives))

# Füttere Algorithmus nur mit den gefilterten Adjektiven
X = texts_transformed
y = df["Score"] >= 4

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

cv = CountVectorizer(max_features = 50)
cv.fit(X_train)

X_train = cv.transform(X_train)
X_test = cv.transform(X_test)

# Lerne mit multinomialen naiven Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# Gebe Adjektive sortiert aus
adj = list(zip(model.coef_[0], cv.get_feature_names()))
adj = sorted(adj)

for i in adj:
    print(i)
