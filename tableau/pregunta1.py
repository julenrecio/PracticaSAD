import pandas as pd
import ast


df = pd.read_csv("../datos/airbnb.csv")
filas_resultado = []

for _, row in df.iterrows():
    try:
        reviews = ast.literal_eval(row['reviews'])
        review_scores = ast.literal_eval(row['review_scores'])
        review_score_value = review_scores.get('review_scores_value', None)
        price = row.get('price', None)

        for review in reviews:
            review_date = review.get('date', None)
            if review_date:
                filas_resultado.append({
                    'review_date': review_date,
                    'review_scores_value': review_score_value,
                    'price': price
                })
    except Exception as e:
        continue

df_final = pd.DataFrame(filas_resultado)
df_final['review_date'] = pd.to_datetime(df_final['review_date'])
df_final.to_csv("../tableau/pregunta1.csv", index=False)

