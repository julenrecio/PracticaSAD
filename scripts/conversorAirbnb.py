import csv
import ast
import sys

if __name__ == "__main__":

    total_expected_reviews = 0
    total_written_reviews = 0
    skipped_no_score = 0
    skipped_empty = 0

    csv.field_size_limit(sys.maxsize)

    input_file = '../datos/airbnb.csv'
    output_file = '../datos/airbnb_simplificado.csv'

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['review', 'score'])

        next(reader)

        for i, row in enumerate(reader, start=2):
            review_scores = row[-6]
            review_scores_dict = ast.literal_eval(review_scores)
            mean_score = review_scores_dict.get('review_scores_value', None)

            if mean_score is None:
                print(f"Fila sin 'review_scores_value' encontrada en la fila {i}")
                skipped_no_score += 1
                continue

            reviews = row[-5]
            reviews_dict = ast.literal_eval(reviews)

            total_expected_reviews += len(reviews_dict)

            for review in reviews_dict:
                review_text = review.get('comments', '').replace('\n', ' ').replace('\r', ' ').strip()
                if review_text:
                    writer.writerow([review_text, int(mean_score)])
                    total_written_reviews += 1
                else:
                    skipped_empty += 1

    print(f"Total esperado: {total_expected_reviews}")
    print(f"Total escrito: {total_written_reviews}")
    print(f"Sin score: {skipped_no_score}")
    print(f"Reviews vacías: {skipped_empty}")
