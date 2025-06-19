import pandas as pd

entrada = "DatosProyecto/tripadvisor_hotel_reviews.csv"
salida = "DatosProyecto/tripadvisor_hotel_reviews_simplificado.csv"
df = pd.read_csv(entrada)
df.rename(columns={"Rating": "score"}, inplace=True)
df.rename(columns={"Review": "review"}, inplace=True)
df["score"] = df["score"].apply(lambda x: round((x - 1) * 2.5))
df.to_csv(salida, index=False)
