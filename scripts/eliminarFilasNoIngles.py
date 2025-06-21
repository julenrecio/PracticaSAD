import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def es_ingles(texto):
    try:
        return detect(str(texto)) == 'en'
    except LangDetectException:
        return False


entrada = "../datos/airbnb_simplificado.csv"
salida = "../datos/airbnb_simplificado_ingles.csv"
df = pd.read_csv(entrada)
df_filtrado = df[df['review'].apply(es_ingles)]
df_filtrado.to_csv(salida, index=False)
print(f"Se han guardado {len(df_filtrado)} reviews en inglés en '{salida}'.")
