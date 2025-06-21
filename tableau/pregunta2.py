import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clasificar_sentimiento(score):
    if pd.isnull(score):
        return 'desconocido'
    elif score >= 9:
        return 'positiva'
    elif score <= 6:
        return 'negativa'
    else:
        return 'neutra'


df = pd.read_csv("../datos/airbnb_simplificado_ingles.csv")
df = df.sample(10000)
df['sentimiento'] = df['score'].apply(clasificar_sentimiento)
filas_palabras = []
stop_words = set(stopwords.words('english'))

for _, row in df.iterrows():
    comentario = row.get("review", "")
    sentimiento = row.get("sentimiento", "desconocido")
    comentario = comentario.lower()
    comentario = re.sub(r"[^\w\s]", "", comentario)
    palabras = word_tokenize(comentario, language='english')
    palabras_filtradas = [p for p in palabras if p not in stop_words and len(p) > 2]

    for palabra in palabras_filtradas:
        filas_palabras.append({
            "palabra": palabra,
            "sentimiento": sentimiento
        })

df_palabras = pd.DataFrame(filas_palabras)
df_palabras.to_csv("pregunta2.csv", index=False)
