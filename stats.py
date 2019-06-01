import sys
import json
import os

def process_stats(file_path, labels):
    with open(file_path, 'r') as source_file:
        source = json.load(source_file)

        stats = {}
        for label in labels:
            stats[label] = 0
        
        for diag in source:
            for item in diag:
                stats[item['emotion']] += 1

        total = 0
        for label in labels:
            total += stats[label]
        
        for label in labels:
            print("{}\t{}\t{}".format(label, stats[label], stats[label]/float(total)))
        
        print('Total = {}'.format(total))

if __name__ == '__main__':

    FILE_PATH = sys.argv[1]

    labels = ['neutral', 'joy', 'sadness', 'anger']
    labels = ["non-neutral", "joy", "sadness", "surprise", "anger", "fear", "disgust", "neutral"]
    process_stats(FILE_PATH, labels)

