import requests
import pandas as pd


def traducir(texto, modelo='tinyllama'):
    prompt = (
        f'If the following text is not in English, translate it to English. '
        f'If it is already in English, return it as-is.\n\n'
        f'Text: "{texto}"'
    )

    respuesta = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": modelo, "prompt": prompt, "stream": False}
    )
    resultado = respuesta.json()
    return resultado.get('response', 'Error: no response')


df_test = pd.read_csv("../datos/airbnb_simplificado.csv")
df_test['review'] = df_test['review'].apply(traducir)
df_test.to_csv("../datos/test.csv", index=False)
