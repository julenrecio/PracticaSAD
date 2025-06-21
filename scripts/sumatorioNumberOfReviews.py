import csv
import sys

if __name__ == "__main__":

    csv.field_size_limit(sys.maxsize)

    input_file = '../datos/airbnb.csv'
    total_reviews = 0

    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            num_reviews = int(row[-18])
            total_reviews += num_reviews

    print(f"Total esperado de reviews según 'number_of_reviews': {total_reviews}")
