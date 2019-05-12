
import json
import os
import sys
import random
from preproccess import tokenize

def writeJson(obj, file_path):
    with open(file_path, 'w') as f:  
        json.dump(obj, f)

def preprocess_train_dev(data_path, file_name, output_dir, do_sanitize=True):
    # create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load original data
    with open(os.path.join(data_path, file_name), 'r') as f:
        source = json.load(f)

    num_utterances = 0
    
    sanitize = lambda str : tokenize(str) if do_sanitize else str
    # Preprocess
    for n, diag in enumerate(source):
        num_utterances += len(diag)
        for item in diag:
            item['utterance'] = sanitize(item['utterance'])
            if 'utterance_de' in item:
                item['utterance_de'] = sanitize(item['utterance_de'])
            if 'utterance_fr' in item:
                item['utterance_fr'] = sanitize(item['utterance_fr'])
            if 'utterance_it' in item:
                item['utterance_it'] = sanitize(item['utterance_it'])
            # item['utterance'] = item['utterance']

    # Split train & dev
    train = []
    train_1 = []
    train_2 = []
    dev = []    
    smaller_dev = []
    num_diag = len(source)
    indeces = list(range(0, num_diag))
    random.seed(1234)
    random.shuffle(indeces)
    dev_end = int(0.1 * num_diag)
    smaller_dev_end = min(dev_end, 5)

    train_1_end = int(dev_end + (num_diag - dev_end) / 2)

    for i in range(0, dev_end):
        dev.append(source[i])

    for i in range(0, smaller_dev_end):
        smaller_dev.append(source[i])

    for i in range(dev_end, num_diag):
        train.append(source[i])

    for i in range(dev_end, train_1_end):
        train_1.append(source[i])
        
    for i in range(train_1_end, num_diag):
        train_2.append(source[i])
        
    # Write output
    train_file_path = os.path.join(output_dir, "{}_{}".format("train", file_name))
    writeJson(train, train_file_path)
    
    train_1_file_path = os.path.join(output_dir, "{}_{}".format("train_1", file_name))
    writeJson(train_1, train_1_file_path)
    
    train_2_file_path = os.path.join(output_dir, "{}_{}".format("train_2", file_name))
    writeJson(train_2, train_2_file_path)

    dev_file_path = os.path.join(output_dir, "{}_{}".format("dev", file_name))
    writeJson(dev, dev_file_path)
    
    smaller_dev_file_path = os.path.join(output_dir, "{}_{}".format("smaller_dev", file_name))
    writeJson(smaller_dev, smaller_dev_file_path)
    
    print('Successfully preprocessed ({} dialogues, {} utterances)'.format(num_diag, num_utterances))
    return train_file_path, dev_file_path, smaller_dev_file_path, train_1_file_path, train_2_file_path

def merge_files(file_path1, file_path2, output_dir, output_file_name):

    with open(file_path1, 'r') as f:
        source1 = json.load(f)

    with open(file_path2, 'r') as f:
        source2 = json.load(f)

    source1.extend(source2)

    with open(os.path.join(output_dir, output_file_name), 'w') as outfile:  
        json.dump(source1, outfile)


def process_EmotionX(friend_data_path, emotionpush_data_path, output_dir, do_sanitize):
    # train & dev
    print("Preprocess train and dev data")
    friends_train_file, friends_dev_file, friends_smaller_dev_file, friends_train_1, friends_train_2 = preprocess_train_dev(friend_data_path, 'friends.augmented.json', output_dir, do_sanitize)
    emotionpush_train_file, emotionpush_dev_file, emotionpush_smaller_dev_file, emotionpush_train_1, emotionpush_train_2 = preprocess_train_dev(emotionpush_data_path, 'emotionpush.augmented.json', output_dir, do_sanitize)

    # Combine Friends & EmotionPush
    print("Combine Friends & EmotionPush data")
    merge_files(friends_train_file, emotionpush_train_file, output_dir, "train.json")
    merge_files(friends_train_1, emotionpush_train_1, output_dir, "train_1.json")
    merge_files(friends_train_2, emotionpush_train_2, output_dir, "train_2.json")
    merge_files(friends_dev_file, emotionpush_dev_file, output_dir, "dev.json")
    merge_files(friends_smaller_dev_file, emotionpush_smaller_dev_file, output_dir, "smaller_dev.json")
    

if __name__ == '__main__':

    DATA_PATH = sys.argv[1]
    friend_data_path = os.path.join(DATA_PATH, 'Friends')
    emotionpush_data_path = os.path.join(DATA_PATH, 'EmotionPush')

    process_EmotionX(
        friend_data_path,
        emotionpush_data_path
        os.path.join(DATA_PATH, "preprocessed"),
        True
        )

    process_EmotionX(
        friend_data_path,
        emotionpush_data_path
        os.path.join(DATA_PATH, "preprocessed_raw"),
        False
        )
