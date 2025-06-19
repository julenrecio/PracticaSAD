import requests
import pandas as pd


def traducir_ollama(texto, modelo='llama3'):
    prompt = f'Traduce al inglés este texto: "{texto}"'
    respuesta = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": modelo, "prompt": prompt}
    )
    resultado = respuesta.json()
    print("Respuesta completa de Ollama:", resultado)
    return resultado['response']


df_test = pd.read_csv("../DatosProyecto/airbnb_simplificado.csv")
df_test['review'] = df_test['review'].apply(traducir_ollama)
df_test.to_csv("../DatosProyecto/test.csv", index=False)
