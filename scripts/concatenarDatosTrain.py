import pandas as pd

df1 = pd.read_csv("../datos/AirBNBReviews_simplificado_NB.csv", header=0)
df2 = pd.read_csv("../datos/tripadvisor_hotel_reviews_simplificado.csv", header=0)
df_unido = pd.concat([df1, df2], ignore_index=True)
df = df_unido.rename(columns={'score': 'review_score'})
df.to_csv("../datos/train.csv", index=False)
