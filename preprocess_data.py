
import json
import os
import sys
import random
from preproccess import tokenize

def writeJson(obj, file_path):
    with open(file_path, 'w') as f:  
        json.dump(obj, f)

def preprocess_train_dev(data_path, file_name, output_dir):
    # create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load original data
    with open(os.path.join(data_path, file_name), 'r') as f:
        source = json.load(f)

    num_utterances = 0
    
    # Preprocess
    for n, diag in enumerate(source):
        num_utterances += len(diag)
        for item in diag:
            item['utterance'] = tokenize(item['utterance'])
            # item['utterance'] = item['utterance']

    # Split train & dev
    train = []
    dev = []    
    smaller_dev = []
    num_diag = len(source)
    indeces = list(range(0, num_diag))
    random.seed(1234)
    random.shuffle(indeces)
    dev_end = int(0.1 * num_diag)
    smaller_dev_end = min(dev_end, 25)

    for i in range(0, dev_end):
        dev.append(source[i])

    for i in range(0, smaller_dev_end):
        smaller_dev.append(source[i])

    for i in range(dev_end, num_diag):
        train.append(source[i])

    # Write output
    train_file_path = os.path.join(output_dir, "{}_{}".format("train", file_name))
    writeJson(train, train_file_path)
    
    dev_file_path = os.path.join(output_dir, "{}_{}".format("dev", file_name))
    writeJson(dev, dev_file_path)
    
    smaller_dev_file_path = os.path.join(output_dir, "{}_{}".format("smaller_dev", file_name))
    writeJson(smaller_dev, smaller_dev_file_path)
    
    print('Successfully preprocessed ({} dialogues, {} utterances)'.format(num_diag, num_utterances))
    return train_file_path, dev_file_path, smaller_dev_file_path

def merge_files(file_path1, file_path2, output_dir, output_file_name):

    with open(file_path1, 'r') as f:
        source1 = json.load(f)

    with open(file_path2, 'r') as f:
        source2 = json.load(f)

    source1.extend(source2)

    with open(os.path.join(output_dir, output_file_name), 'w') as outfile:  
        json.dump(source1, outfile)


if __name__ == '__main__':

    DATA_PATH = sys.argv[1]
    friend_data_path = os.path.join(DATA_PATH, 'Friends')
    emotionpush_data_path = os.path.join(DATA_PATH, 'EmotionPush')
    output_dir = os.path.join(DATA_PATH, "preprocessed")

    # train & dev
    print("Preprocess train and dev data")
    friends_train_file, friends_dev_file, friends_smaller_dev_file = preprocess_train_dev(friend_data_path, 'friends.json', output_dir)
    emotionpush_train_file, emotionpush_dev_file, emotionpush_smaller_dev_file = preprocess_train_dev(emotionpush_data_path, 'emotionpush.json', output_dir)

    # Combine Friends & EmotionPush
    print("Combine Friends & EmotionPush data")
    merge_files(friends_train_file, emotionpush_train_file, output_dir, "train.json")
    merge_files(friends_dev_file, emotionpush_dev_file, output_dir, "dev.json")
    merge_files(friends_smaller_dev_file, emotionpush_smaller_dev_file, output_dir, "smaller_dev.json")
    