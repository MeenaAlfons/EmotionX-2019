from __future__ import absolute_import, division, print_function

import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule #, warmup_linear

from utils import compute_metrics, convert_examples_to_features

class Trainer(object):

    def __init__(self, args):
        self.args = args

    def save_model(self, model, tokenizer, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save a trained model, configuration and tokenizer
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)


    def evaluate(self, print_preds):
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num examples = %d", len(self.eval_examples))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if self.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            elif self.output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            nb_eval_examples += input_ids.size(0)
            logits = logits.detach().cpu().numpy()
            
            if len(preds) == 0:
                preds.append(logits)
            else:
                preds[0] = np.append(
                    preds[0], logits, axis=0)

        eval_loss = eval_loss / nb_eval_steps
        eval_loss_examples = eval_loss / nb_eval_examples
        preds = preds[0]
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds)
        
        eval_all_label_ids_numpy = self.eval_all_label_ids.numpy()
        result = compute_metrics("many_metrics", preds, eval_all_label_ids_numpy)

        if print_preds:
            for i, pred in enumerate(preds):
                print("i = {}\t\tPredicted = {}\t\tActual = {}".format(i, pred, eval_all_label_ids_numpy[i]))

        result['eval_loss'] = eval_loss
        result['eval_loss_examples'] = eval_loss_examples
        return result, preds

    def run(self):
        self.logger.info("***** Running *****")
        self.logger.info("  Num examples = %d", len(self.run_examples))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()
        preds = []

        for input_ids, input_mask, segment_ids in tqdm(self.run_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

            logits = logits.detach().cpu().numpy()
            
            if len(preds) == 0:
                preds.append(logits)
            else:
                preds[0] = np.append(
                    preds[0], logits, axis=0)

        preds = preds[0]
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds)
    
        return preds

    def save_result(self, result, output_eval_dir):
        output_eval_file = os.path.join(output_eval_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            self.logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def preprare_distant_debugging(self):
        if self.args.server_ip and self.args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.args.server_ip, self.args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()
            
    def prepare_device(self):
        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

    def prepare_logging(self):
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)

        self.logger = logging.getLogger(__name__)
        self.logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.args.local_rank != -1), self.args.fp16))

    def seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def prepare_model(self):
        model_dir = self.args.resume_dir if self.args.resume_dir else self.args.bert_model
        cache_dir = self.args.cache_dir if self.args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(self.args.local_rank))

        self.model = BertForSequenceClassification.from_pretrained(model_dir,
                                                                   cache_dir=cache_dir,
                                                                   num_labels=self.num_labels)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=self.args.do_lower_case)
        
        if self.args.fp16:
            self.model.half()
            
        print(self.device)
            
        self.model.to(self.device)
        
        if self.args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            self.model = DDP(self.model)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def prepare_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if self.args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            self.optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=self.args.learning_rate,
                                bias_correction=False,
                                max_grad_norm=1.0)
            self.warmup_schedule = WarmupLinearSchedule(warmup=self.args.warmup_proportion, t_total=self.num_train_optimization_steps)
            
            if self.args.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.args.loss_scale)

        else:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=self.args.learning_rate,
                                warmup=self.args.warmup_proportion,
                                t_total=self.num_train_optimization_steps,
                                weight_decay=0.01)
        
    def prepare_train_examples(self):
        self.train_examples = self.processor.get_train_examples(self.args.data_dir)
        self.num_train_optimization_steps = int(
            len(self.train_examples) / self.args.train_batch_size / self.args.gradient_accumulation_steps) * self.args.num_train_epochs
        if self.args.local_rank != -1:
            self.num_train_optimization_steps = self.num_train_optimization_steps // torch.distributed.get_world_size()
        
        weights, augmented_weights = self.processor.get_train_weights()
        self.label_weights = augmented_weights
        print("label_weights = {}".format(self.label_weights))
    
        input_length_arr = []
        if self.processor.is_pair():
            truncate_seq_pair = lambda tokens_a, tokens_b, max_length : self.processor.truncate_seq_pair(tokens_a, tokens_b, max_length)
            train_features = convert_examples_to_features(
                self.train_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr,
                truncate_seq_pair=truncate_seq_pair)
        else:
            train_features = convert_examples_to_features(
                self.train_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr)
            
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        input_length_arr = np.array(input_length_arr)
        print("Train input_length_arr: max={}, min={}, avg={}".format(np.max(input_length_arr),
                                                                      np.min(input_length_arr),
                                                                      np.mean(input_length_arr)))
        
        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

    def preprare_eval_examples(self):
        self.eval_examples = self.processor.get_dev_examples(self.args.data_dir)
        
        input_length_arr = []
        if self.processor.is_pair():
            truncate_seq_pair = lambda tokens_a, tokens_b, max_length : self.processor.truncate_seq_pair(tokens_a, tokens_b, max_length)
            self.eval_features = convert_examples_to_features(
                self.eval_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr,
                truncate_seq_pair=truncate_seq_pair)
        else:
            self.eval_features = convert_examples_to_features(
                self.eval_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr)
            
        all_input_ids = torch.tensor([f.input_ids for f in self.eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.eval_features], dtype=torch.long)

        
        input_length_arr = np.array(input_length_arr)  
        print("Eval input_length_arr: max={}, min={}, avg={}".format(np.max(input_length_arr),
                                                                     np.min(input_length_arr),
                                                                     np.mean(input_length_arr)))
        
        if self.output_mode == "classification":
            self.eval_all_label_ids = torch.tensor([f.label_id for f in self.eval_features], dtype=torch.long)
        elif self.output_mode == "regression":
            self.eval_all_label_ids = torch.tensor([f.label_id for f in self.eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, self.eval_all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

    def prepare_run_examples(self):
        self.run_examples = self.processor.get_dev_examples(self.args.data_dir)
        
        input_length_arr = []
        if self.processor.is_pair():
            truncate_seq_pair = lambda tokens_a, tokens_b, max_length : self.processor.truncate_seq_pair(tokens_a, tokens_b, max_length)
            self.run_features = convert_examples_to_features(
                self.run_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr,
                truncate_seq_pair=truncate_seq_pair)
        else:
            self.run_features = convert_examples_to_features(
                self.run_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode,
                self.logger,
                input_length_arr)
            
        all_input_ids = torch.tensor([f.input_ids for f in self.run_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.run_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.run_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.run_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)


    def train(self):
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.train_examples))
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        self.logger.info("  Num steps = %d", self.num_train_optimization_steps)

        
        if self.output_mode == "classification":
            print("label_weights = {}".format(self.label_weights))
            label_weights = torch.tensor(self.label_weights).float().to(self.device)
            loss_fct = CrossEntropyLoss(weight=label_weights)
        elif self.output_mode == "regression":
            loss_fct = MSELoss()

        self.model.train()
        
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            self.model.train()

            if epoch == self.args.resume_epochs - 1:
                time.sleep(1)
                tqdm.write("\nEpoch {} previously done\n".format(epoch))

            if epoch < self.args.resume_epochs:    
                continue
            elif epoch == self.args.resume_epochs:
                tqdm.write("\nResuming Epoch {}.\n".format(epoch))

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
#             start_time = time.time()
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
#                 print("Reading Batch in {}".format(time.time() - start_time))

                if epoch == self.args.resume_epochs and step == self.args.resume_steps - 1:
                    time.sleep(1)
                    tqdm.write("\nStep {} of epoch {} previously done\n".format(step, epoch))

                if epoch == self.args.resume_epochs:
                    if step < self.args.resume_steps:
                        continue
                    elif step == self.args.resume_steps:
                        tqdm.write("\nResuming step {} from epoch {}.".format(step, epoch))

#                 start_time = time.time()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
#                 print("Preparing Batch in {}".format(time.time() - start_time))

                # define a new function to compute loss values for both output_modes
#                 start_time = time.time()
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)
#                 print("Execute model in {}".format(time.time() - start_time))

#                 start_time = time.time()
                if self.output_mode == "classification":
                    loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.output_mode == "regression":
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
#                 print("Calculate loss in {}".format(time.time() - start_time))

#                 start_time = time.time()
                if self.args.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
#                 print("Backword loss in {}".format(time.time() - start_time))

#                 start_time = time.time()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if self.args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = self.args.learning_rate * warmup_schedule.get_lr(global_step/self.num_train_optimization_steps)
#                         lr_this_step = self.args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, self.args.warmup_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
#                 print("Optimize in {}".format(time.time() - start_time))

                if (step+1) % self.args.save_model_steps == 0:
                    # Save model
                    # ...
                    step_output_dir = os.path.join(self.args.output_dir, "epoch_{}_step_{}".format(epoch, step))
                    self.save_model(self.model, self.tokenizer, step_output_dir)

#                 start_time = time.time()

            # Save model at the end of Epoch
            # ...
            epoch_output_dir = os.path.join(self.args.output_dir, "epoch_{}".format(epoch))
            self.save_model(self.model, self.tokenizer, epoch_output_dir)
            
            # Evaluate Epoch
            result, _ = self.evaluate(False)
            result['tr_loss'] = tr_loss/nb_tr_steps
            result['tr_loss_examples'] = tr_loss/nb_tr_examples
            self.save_result(result, epoch_output_dir)
            
        
    def execute(self):
        self.preprare_distant_debugging()

        self.prepare_device()

        self.prepare_logging()

        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                                self.args.gradient_accumulation_steps))

        self.args.train_batch_size = self.args.train_batch_size // self.args.gradient_accumulation_steps

        self.seed()

        if not self.args.do_train and not self.args.do_eval and not self.args.do_run:
            raise ValueError("At least one of `do_train`, `do_eval` or `do_run` must be True.")

        if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir) and self.args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.args.output_dir))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.processor = self.args.processor()
        self.output_mode = self.args.output_mode

        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)

        self.prepare_model()

        if self.args.do_train:
            self.prepare_train_examples()
            self.prepare_optimizer()

        if self.args.do_train or self.args.do_eval:
            self.preprare_eval_examples()

        if self.args.do_train:
            self.train()

        if self.args.do_eval and (self.args.local_rank == -1 or torch.distributed.get_rank() == 0):
            result, _ = self.evaluate(False)
            self.save_result(result, args.output_dir)

        if self.args.do_run:
            self.prepare_run_examples()
            preds = self.run()
            self.processor.save_dev(self.args.data_dir, self.run_examples, preds)

