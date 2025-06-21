import pandas as pd

df = pd.read_csv("../datos/train.csv", header=0)
conteo = df["review_score"].value_counts()
clases_validas = conteo[conteo >= 5].index
df_clases_validas = df[df["review_score"].isin(clases_validas)].reset_index(drop=True)
df_clases_validas.to_csv("../datos/train.csv", index=False)
