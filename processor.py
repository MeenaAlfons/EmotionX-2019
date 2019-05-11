import os
import json

import numpy as np

from utils import InputExample, DataProcessor

class EmotionX2019Processor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.json"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_labels(self):
        """See base class."""
        return ["non-neutral", "joy", "sadness", "surprise", "anger", "fear", "disgust", "neutral"]
      
    def get_weights(self):
        labels = self.get_labels()
        weights = []
        total = 0
        for i, label in enumerate(labels):
            total += self.stats[label]
            weights.append(self.stats[label])
        
        print("weights list = {}".format(weights))
        weights = np.array(weights, np.double)
        weights = total - weights
        weights = weights / max(weights)
        print("Final weights = {}".format(weights))
        return weights
    
    def is_pair(self):
        return False
      
    def _create_examples(self, file_name, set_type):
        """Creates examples for the training and dev sets."""   
        print("Creating Examples ...................")
        stats = {}
        labels = self.get_labels()
        for label in labels:
            stats[label] = 0
        
        examples = []     
        with open(file_name) as file:
            source = json.load(file)
            for (i, diag) in enumerate(source):
                for (j, item) in enumerate(diag):
                    guid = "{}-{}-{}".format(set_type, i, j)
                    example = self._create_example_of_item(guid, diag, j);
                    examples.append(example)
                    stats[example.label] += 1
                    
        print("stats = {}".format(stats))
        self.stats = stats
        return examples
      
    def _create_example_of_item(self, guid, diag, i):
        raise NotImplementedError()


class EmotionX2019_OneSequence_Processor(EmotionX2019Processor):
    """Processor for the RTE data set (GLUE version)."""

    def _create_example_of_item(self, guid, diag, i):
        item = diag[i]
        text_a = item["utterance"]
        label = item["emotion"]
        return InputExample(guid=guid, text_a=text_a, label=label)


class EmotionX2019_TwoSequence_Processor(EmotionX2019Processor):
    """Processor for the RTE data set (GLUE version)."""

    def is_pair(self):
        return True
    
    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        pass
        


class EmotionX2019_NextSameSpeaker_Processor(EmotionX2019_TwoSequence_Processor):
    
    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_b) > 0:
                tokens_b.pop()
            else:
                tokens_a.pop()
            
        
    def _create_example_of_item(self, guid, diag, i):
        item = diag[i]
        text_a = item["utterance"]
        label = item["emotion"]
        text_b=None
        j = i+1
        while j < len(diag) :
            if(diag[j]["speaker"] == item["speaker"]):
                text_b = diag[j]["utterance"]
                break
            j = j + 1
#         print("text_a={}".format(text_a))
#         print("text_b={}".format(text_b))
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

class EmotionX2019_PrevSameSpeaker_Processor(EmotionX2019_TwoSequence_Processor):
    """Processor for the RTE data set (GLUE version)."""

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
                
            if len(tokens_a) > 0:
                tokens_a.pop()
            else:
                tokens_b.pop()
            
    def _prevSameSpeakerItem(self, diag, i):
        origItem = diag[i]
        j = i-1
        while j>=0:
            currentItem = diag[j]
            if currentItem["speaker"] == origItem["speaker"]:
                return currentItem
            j = j - 1
        return None
    
    def _create_example_of_item(self, guid, diag, i):
        item = diag[i]
        text_b = item["utterance"]
        label = item["emotion"]
        text_a=''
        
        prevItem = self._prevSameSpeakerItem(diag, i)
        if prevItem:
            text_a = prevItem["utterance"]
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

class EmotionX2019_PrevDiffSpeaker_Processor(EmotionX2019_TwoSequence_Processor):
    """Processor for the RTE data set (GLUE version)."""

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
                
            if len(tokens_a) > 0:
                tokens_a.pop()
            else:
                tokens_b.pop()
            
    def _prevDiffSpeakerItem(self, diag, i):
        origItem = diag[i]
        j = i-1
        while j>=0:
            currentItem = diag[j]
            if currentItem["speaker"] != origItem["speaker"]:
                return currentItem
            j = j - 1
        return None
    
    def _create_example_of_item(self, guid, diag, i):
        item = diag[i]
        text_b = item["utterance"]
        label = item["emotion"]
        text_a=''
        
        prevItem = self._prevDiffSpeakerItem(diag, i)
        if prevItem:
            text_a = prevItem["utterance"]
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)