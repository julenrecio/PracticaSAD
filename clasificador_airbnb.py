# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import pickle
import sys
import time
import unicodedata
from datetime import datetime

import matplotlib.pyplot as plt
# Nltk
import nltk
import numpy as np
import pandas as pd
# Colorama
from colorama import Fore
# Imblearn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# Sklearn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-a", "--algorithm",
                       help="Algoritmo a utilizar (kNN, decision_tree, random_forest, naive_bayes)", required=False,
                       default="kNN")
    parse.add_argument("-f", "--features", help="Número mmáximo de columnas del modelo", required=False,
                       default=1000, type=int)
    parse.add_argument("-e", "--estimator",
                       help="Estimador a utilizar para elegir el mejor modelo "
                            "https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter",
                       required=False, default="f1_micro")

    arguments = parse.parse_args()
    with open('clasificador_airbnb.json') as json_file:
        config = json.load(json_file)

    for key, value in config.items():
        setattr(arguments, key, value)

    return arguments


def calculate_fscore(y_test, y_pred):
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro


def calculate_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, zero_division=0)
    return report


def calculate_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm


def simplify_text(data):
    try:
        col = 'review'
        data = data[data[col].notnull()]
        data.reset_index(drop=True, inplace=True)
        data[col] = data[col].apply(lambda x: x.lower())
        data[col] = data[col].apply(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
        data[col] = data[col].apply(lambda x: [word for word in x if not word.isnumeric()])
        data[col] = data[col].apply(
            lambda x: [word for word in x if word not in nltk.corpus.stopwords.words('english')])
        data[col] = data[col].apply(lambda x: [WordNetLemmatizer().lemmatize(word) for word in x])
        data[col] = data[col].apply(lambda x: ''.join(
            patata for patata in unicodedata.normalize('NFD', ' '.join(x)) if
            unicodedata.category(patata) != 'Mn'))
        return data

    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_text(data):
    try:
        text_col = data["review"]
        v = TfidfVectorizer(max_features=args.features)
        text_data = text_col.astype(str)
        x = v.fit_transform(text_data)
        df1 = pd.DataFrame.sparse.from_spmatrix(x, columns=v.get_feature_names_out())
        data.drop("review", axis=1, inplace=True)
        data = pd.concat([data.reset_index(drop=True), df1.reset_index(drop=True)], axis=1)
        # data.to_csv('output/data-processed.csv', index=False)
        return data
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def divide_data(data):
    y = data["review_score"]
    x = data.drop(columns=["review_score"])
    # x.sort_index(axis=1, inplace=True)

    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.4, stratify=y, random_state=42)
    return x_train, x_dev, y_train, y_dev


def save_model(gs):
    try:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f'output/modelo_{args.algorithm}_{args.features}_{timestamp}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)
        file.close()
        with open(f'output/modelo_{args.algorithm}_{args.features}_{timestamp}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
        file.close()

        data = []
        with open(f'output/modelo_{args.algorithm}_{args.features}_{timestamp}.csv', "r") as file:
            next(file)
            for line in file:
                if line.strip() == "":
                    continue
                line_data = [item.strip() for item in line.strip().split('",')]
                params = line_data[0]
                score = float(line_data[1])
                data.append((params, score))

        params = [param for param, _ in data]
        scores = [score for _, score in data]

        plt.figure(figsize=(10, 6))
        plt.bar(params, scores, color='skyblue', edgecolor='black')
        plt.xticks(rotation=90)
        plt.xlabel('Hiperparámetros')
        plt.ylabel('Score')
        plt.title('Hiperparámetros y su score')
        plt.savefig(f'output/scores_{args.algorithm}_{args.features}_{timestamp}.png', bbox_inches='tight')

    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)


def mostrar_resultados(gs, x_dev, y_dev):
    resultados_df = pd.DataFrame(gs.cv_results_)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', 100)
    print(resultados_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    print(Fore.MAGENTA + "> Mejores parametros:\n" + Fore.RESET, gs.best_params_)
    print(Fore.MAGENTA + "> Mejor puntuacion:\n" + Fore.RESET, gs.best_score_)
    print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
    print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
    print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET,
          calculate_classification_report(y_dev, gs.predict(x_dev)))
    print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET,
          calculate_confusion_matrix(y_dev, gs.predict(x_dev)))


def mostrar_resultados_test(y_test, y_pred):
    print("> Informe de clasificación:\n", calculate_classification_report(y_test, y_pred))
    print("> Matriz de confusión:\n", calculate_confusion_matrix(y_test, y_pred))
    print("> F1-score micro:\n", calculate_fscore(y_test, y_pred)[0])
    print("> F1-score macro:\n", calculate_fscore(y_test, y_pred)[1])


def algorithm(classiffier, data):
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data(data)

    algo = args.algorithm
    param_grid = getattr(args, algo)

    # Hacemos un barrido de hiperparametros
    gs = GridSearchCV(classiffier, param_grid, cv=5, scoring=args.estimator, verbose=1)
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)


def load_model():
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model_pickle = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model_pickle
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)


def predict():
    global file_data
    columnas = file_data.columns
    columnasmodelo = model.feature_names_in_
    for i in range(len(columnas)):
        if columnas[i] not in columnasmodelo:
            file_data.drop(columnas[i], axis=1, inplace=True)
    for j in range(len(columnasmodelo)):
        if columnasmodelo[j] not in columnas:
            file_data = pd.concat([file_data, pd.DataFrame([0] * len(file_data), columns=[columnasmodelo[j]])], axis=1)
    # data.sort_index(axis=1, inplace=True)
    pd.DataFrame(model.feature_names_in_, columns=['Columnas en modelo']).to_csv('output/modelColumnas.csv',
                                                                                 index=False)
    y_pred = model.predict(file_data)
    y_test = pd.read_csv('output/y_test.csv')

    mostrar_resultados_test(y_test, y_pred)

    # Añadimos la prediccion al dataframe data
    file_data = pd.concat([file_data, pd.DataFrame(y_pred, columns=["review_score"])], axis=1)


if __name__ == "__main__":

    np.random.seed(42)
    args = parse_args()
    try:
        os.makedirs('output')
    except FileExistsError:
        pass

    if args.mode == "train":
        file_data = pd.read_csv("DatosProyecto/train.csv", encoding='utf-8')
        file_data = simplify_text(file_data)
        file_data = process_text(file_data)
        print("Datos procesados")

        classifier = KNeighborsClassifier()
        if args.algorithm == "kNN":
            classifier = KNeighborsClassifier()
        elif args.algorithm == "decision_tree":
            classifier = DecisionTreeClassifier(random_state=42)
        elif args.algorithm == "random_forest":
            classifier = RandomForestClassifier(random_state=42)
        elif args.algorithm == "naive_bayes":
            classifier = MultinomialNB()
        try:
            algorithm(classifier, file_data)
            print("Algoritmo ejecutado con éxito")
            sys.exit(0)
        except Exception as ex:
            print(ex)

    elif args.mode == "test":
        file_data = pd.read_csv("DatosProyecto/test.csv", encoding='utf-8')
        model = load_model()
        try:
            predict()
            file_data.to_csv('output/data-prediction.csv', index=False)
            sys.exit(0)
        except Exception as ex:
            print(ex)
            sys.exit(1)
    else:
        print(Fore.RED + "Modo no soportado" + Fore.RESET)
        sys.exit(1)
