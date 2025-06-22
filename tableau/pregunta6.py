import pandas as pd
import ast

df = pd.read_csv("../datos/airbnb.csv")
df["availability_365"] = df["availability"].apply(
    lambda x: ast.literal_eval(x).get("availability_365") if pd.notnull(x) else None
)
df["review_scores_value"] = df["review_scores"].apply(
    lambda x: ast.literal_eval(x).get("review_scores_value") if pd.notnull(x) else None
)
columnas_deseadas = ["price", "security_deposit", "cleaning_fee", "availability_365", "review_scores_value"]
df = df[columnas_deseadas]

for col in columnas_deseadas:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    media = df.loc[df[col] > 0, col].mean()
    df[col] = df[col].apply(lambda x: media if pd.isna(x) or x == 0 else x)

df.to_csv("pregunta6.csv", index=False)
