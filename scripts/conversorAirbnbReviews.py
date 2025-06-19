from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

entrada = "../DatosProyecto/AirBNBReviews.csv"
salida = "../DatosProyecto/AirBNBReviews_simplificado_NB.csv"
df = pd.read_csv(entrada)
df.drop(df.columns[0], axis=1, inplace=True)
df.dropna(how='all', inplace=True)
df.rename(columns={"Positive or Negative": "score"}, inplace=True)
df.rename(columns={"Review": "review"}, inplace=True)

vectorizer = TfidfVectorizer(max_features=1000)
texto = df['review']
x = vectorizer.fit_transform(texto)
y = df['score']

modelo = MultinomialNB()
modelo.fit(x, y)
probs = modelo.predict_proba(x)[:, 1]
pred_0_10 = (probs * 10).round().astype(int)

df['score'] = pred_0_10
df.to_csv(salida, index=False)
