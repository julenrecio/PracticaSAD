import pandas as pd
import ast

df = pd.read_csv("../datos/airbnb.csv")
df["latitude"] = None
df["longitude"] = None
df["score"] = None

for idx, row in df.iterrows():
    try:
        address = ast.literal_eval(row["address"])
        coords = address["location"]["coordinates"]
        df.at[idx, "longitude"] = coords[0]
        df.at[idx, "latitude"] = coords[1]
        review_scores = ast.literal_eval(row['review_scores'])
        review_score_value = review_scores.get('review_scores_value', 9)  # Medía = 9.429 ---> Redondeo a 9
        df.at[idx, "score"] = review_score_value
    except Exception:
        continue

columnas_deseadas = ["latitude", "longitude", "price", "score"]
columnas_finales = [col for col in columnas_deseadas if col in df.columns]
df_final = df[columnas_finales]
df_final.to_csv("pregunta4.csv", index=False)
