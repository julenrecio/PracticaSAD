import pandas as pd
from tabulate import tabulate


def visualizar_csv(ruta_csv, n_filas=150):
    try:
        df = pd.read_csv(ruta_csv)

        print(f"\n✅ Archivo cargado correctamente: '{ruta_csv}'")
        print(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

        print(f"\n🔎 Mostrando las primeras {n_filas} filas:\n")
        print(tabulate(df.head(n_filas), headers='keys', tablefmt='grid'))

        # print("\n📘 Tipos de datos por columna:")
        # print(tabulate(df.dtypes.reset_index().values, headers=["Columna", "Tipo"], tablefmt="fancy_grid"))

        col = df[df.columns[0]]
        print("Min:", col.min())
        print("Max:", col.max())
        print("Contiene 0:", (col == 0).any())
        print("Contiene 1:", (col == 1).any())


    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_csv}'.")
    except pd.errors.ParserError:
        print(f"❌ Error: El archivo '{ruta_csv}' no se pudo analizar correctamente.")
    except Exception as e:
        print(f"⚠️ Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    ruta = input("Introduce la ruta del archivo CSV: ")
    visualizar_csv(ruta)
