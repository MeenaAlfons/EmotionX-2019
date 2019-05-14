
import json
import csv
import os
import sys
import random
from preprocess_data import preprocess_train_dev

def process_eval(friend_data_path, emotionpush_data_path, output_dir, do_sanitize):
    # train & dev
    print("Preprocess train and dev data")
    friends_train_file, friends_dev_file, friends_smaller_dev_file, friends_train_1, friends_train_2 = preprocess_train_dev(friend_data_path, 'friends_eval.json', output_dir, do_sanitize)
    emotionpush_train_file, emotionpush_dev_file, emotionpush_smaller_dev_file, emotionpush_train_1, emotionpush_train_2 = preprocess_train_dev(emotionpush_data_path, 'emotionpush_eval.json', output_dir, do_sanitize)
    

if __name__ == '__main__':

    DATA_PATH = sys.argv[1]
    friend_data_path = DATA_PATH
    emotionpush_data_path = DATA_PATH

    process_eval(
        friend_data_path,
        emotionpush_data_path,
        os.path.join(DATA_PATH, "preprocessed"),
        True
        )
