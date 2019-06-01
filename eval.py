import json
import sys
from utils import many_metrics, convert_examples_to_features


def process_eval(source_file_path, pred_file_path):
    with open(source_file_path, 'r') as source_file, open(pred_file_path, 'r') as pred_file:
        source_json = json.load(source_file)
        pred_json = json.load(pred_file)
        
        y_true = []
        y_pred = []
        for source_diag, pred_diag in zip(source_json, pred_json):
            for source_item, pred_item in zip(source_diag, pred_diag):
                y_true.append(source_item['emotion'])
                y_pred.append(pred_item['emotion'])

        result = many_metrics(y_pred, y_true)

        print('Result = {}'.format(result))

if __name__ == '__main__':

    SOURCE_FILE_PATH = sys.argv[1]
    PRED_FILE_PATH = sys.argv[2]

    process_eval(SOURCE_FILE_PATH, PRED_FILE_PATH)

