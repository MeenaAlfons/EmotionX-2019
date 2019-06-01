import sys
import json
import os

def process_result(data_path, source_file_name, result_file_name):
    with open(os.path.join(data_path, source_file_name), 'r') as source_file:
        source = json.load(source_file)

        dialogs = []
        for diag in source:
            newDiag = []
            for item in diag:
                newItem = {}
                newItem['speaker'] = item['speaker']
                newItem['utterance'] = item['origin']
                newItem['emotion'] = item['emotion']
                newDiag.append(newItem)
            dialogs.append(newDiag)
        
        result = []
        result.append({
            "name": "Meena Alfons",
            "email": "me@meenaalfons.com"
        })
        result.append(dialogs)

        with open(os.path.join(data_path, result_file_name), 'w') as result_file:
            json.dump(result, result_file, indent=4)
    
if __name__ == '__main__':

    DATA_PATH = sys.argv[1]

    process_result(DATA_PATH, 'result_friends_others_dev.json', 'friends_pred.json')
    process_result(DATA_PATH, 'result_emotionpush_others_dev.json', 'emotionpush_pred.json')

