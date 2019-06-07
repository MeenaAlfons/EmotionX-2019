# EmotionX-2019

## Abstract

In this paper, we present a solution for the Dialogue Emotion Recognition Challenge, EmotionX-2019 based onBidirectional Encoder Representations from Transformer (BERT) which is the state-of-the-art for Natural Language Processing with fine-tuning on dialogue utterance classification. We use cascade classification to tackle the dominance of a majority class present in the data. Cascading the classifiers  allowed  to improve our reported accuracy measures for emotion prediction in text.

# How To Run

## Install Dependencies

Use the following commands to install dependencies

```sh
git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
git checkout master
python setup.py install
pip install ./
cd ..

pip install emoji

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```

If you are using Colab, use the following:

```
!git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
```
```
import os
os.chdir("pytorch-pretrained-BERT")
```
```
!git checkout master
!python setup.py install
!pip install ./
```
```
import os
os.chdir("..")
```
```
!git clone https://github.com/NVIDIA/apex
```
```
import os
os.chdir("apex")
```
```
!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
```
import os
os.chdir("..")
```
```
!pip install emoji
```

## Clone the repository

```sh
git clone https://github.com/MeenaAlfons/EmotionX-2019.git
cd EmotionX-2019
```
Or using Colab:

```
!git clone https://github.com/MeenaAlfons/EmotionX-2019.git
```
```
import os
os.chdir("EmotionX-2019")
```

## Extract datasets

Extract the EmotionPush and Friends datasets in the following heirarchy:
```
_ . (current directory)
 |_ dataset
   |_ EmotionPush
   | |_ emotionpush.augmented.json
   | |_ emotionpush.json
   |_ Friends
     |_ friends.augmented.json
     |_ friends.json
``` 
You may use the following commands:
```
unzip -o ./path/to/emotionpush.zip -d ./dataset
unzip -o ./path/to/friends.zip -d ./dataset
```

## Run Preprocessing

```
python EmotionX-2019/preprocess_data.py ./dataset
```

## Train Solution

**Train Friends Majory Classifier**

```python
from processor import Majority_OneSentence_Processor
from Trainer import Trainer

class Args(object):
    pass
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './dataset/preprocessed/'
args.train_file = 'train_friends.augmented.json'
args.dev_file = 'dev_friends.augmented.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = True
args.do_eval = False
args.do_run = False
args.num_train_epochs = 8.0
args.max_seq_length = 256
args.processor = Majority_OneSentence_Processor
args.output_dir =  os.path.join(model_dir, 'friends_majority')
args.resume_dir = None

args.learning_rate = 1e-5
args.seed = 1991

trainer = Trainer(args)
trainer.execute()
```

**Train EmotionPush Majory Classifier**

```python
from processor import Majority_OneSentence_Processor
from Trainer import Trainer

class Args(object):
    pass
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './dataset/preprocessed/'
args.train_file = 'train_emotionpush.augmented.json'
args.dev_file = 'dev_emotionpush.augmented.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = True
args.do_eval = False
args.do_run = False
args.num_train_epochs = 8.0
args.max_seq_length = 256
args.processor = Majority_OneSentence_Processor
args.output_dir =  os.path.join(model_dir, 'emotionpush_majority')
args.resume_dir = None

args.learning_rate = 1e-5
args.seed = 1991

trainer = Trainer(args)
trainer.execute()
```

**Train Friends Others Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.data_dir = './dataset/preprocessed/'
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './dataset/preprocessed/'
args.train_file = 'train_friends.augmented.json'
args.dev_file = 'dev_friends.augmented.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = True
args.do_eval = False
args.do_run = False
args.num_train_epochs = 8.0
args.max_seq_length = 256
args.processor = Others_OneSentence_Processor
args.output_dir = os.path.join(model_dir, 'friends_others')
args.resume_dir = None

args.learning_rate = 1e-5
args.seed = 1991

trainer = Trainer(args)
trainer.execute()
```

**Train EmotionPush Others Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.data_dir = './dataset/preprocessed/'
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './dataset/preprocessed/'
args.train_file = 'train_emotionpush.augmented.json'
args.dev_file = 'dev_emotionpush.augmented.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = True
args.do_eval = False
args.do_run = False
args.num_train_epochs = 8.0
args.max_seq_length = 256
args.processor = Others_OneSentence_Processor
args.output_dir = os.path.join(model_dir, 'emotionpush_others')
args.resume_dir = None

args.learning_rate = 1e-5
args.seed = 1991

trainer = Trainer(args)
trainer.execute()
```

### Run Solution

**Run Friends Majority Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './eval/'
args.train_file = None
args.dev_file = 'friends_eval.json'
args.result_file = 'friends_majority_result.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = False
args.do_eval = False
args.do_run = True
args.num_train_epochs = 1.0
args.max_seq_length = 256
args.processor = Majority_OneSentence_Processor
args.output_dir = None
args.resume_dir = os.path.join(model_dir, 'friends_majority/epoch_0')

args.learning_rate = 1e-5
args.seed = 69847

args.included_labels=2

trainer = Trainer(args)
trainer.execute()
```

**Run Friends Others Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.data_dir = './dataset/preprocessed/'
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './eval/'
args.train_file = None
args.dev_file = 'friends_majority_result.json'
args.result_file = 'friends_result.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = False
args.do_eval = False
args.do_run = True
args.num_train_epochs = 1.0
args.max_seq_length = 256
args.processor = Others_OneSentence_Processor
args.output_dir = None
args.resume_dir = os.path.join(model_dir, 'friends_others/epoch_1')

args.learning_rate = 1e-5
args.seed = 69847

# include first 3 labels (joy, sadness, anger)
args.included_labels=3

trainer = Trainer(args)
trainer.execute()
```

**Run EmotionPush Majority Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './eval/'
args.train_file = None
args.dev_file = 'emotionpush_eval.json'
args.result_file = 'emotionpush_majority_result.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = False
args.do_eval = False
args.do_run = True
args.num_train_epochs = 1.0
args.max_seq_length = 256
args.processor = Majority_OneSentence_Processor
args.output_dir = None
args.resume_dir = os.path.join(model_dir, 'emotionpush_majority/epoch_0')

args.learning_rate = 1e-5
args.seed = 69847

# labels are (yes, no)
args.included_labels=2

trainer = Trainer(args)
trainer.execute()
```

**Run EmotionPush Others Classifier**

```python
args = Args()

args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.warmup_proportion = 0.1
args.cache_dir = "./cache"
args.no_cuda = False
args.local_rank = -1
args.fp16 = False
args.loss_scale = 0
args.gradient_accumulation_steps = 1
args.server_ip = ''
args.server_port = ''
args.output_mode = "classification"
args.data_dir = './dataset/preprocessed/'
args.save_model_steps = 2000
args.resume_epochs = 0
args.resume_steps = 0

# Important configurations
args.data_dir = './eval/'
args.train_file = None
args.dev_file = 'emotionpush_majority_result.json'
args.result_file = 'emotionpush_result.json'
args.train_batch_size = 32
args.eval_batch_size = 32
args.do_train = False
args.do_eval = False
args.do_run = True
args.num_train_epochs = 1.0
args.max_seq_length = 256
args.processor = Others_OneSentence_Processor
args.output_dir = None
args.resume_dir = os.path.join(model_dir, 'emotionpush_others/epoch_1')

args.learning_rate = 1e-5
args.seed = 69847

# include first 3 labels (joy, sadness, anger)
args.included_labels=3

trainer = Trainer(args)
trainer.execute()
```
