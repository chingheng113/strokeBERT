# Code is adapted from the PyTorch pretrained BERT repo - See copyright & license below. 

# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import pickle
import random
import sys


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_curve, auc
#added
import json
from random import shuffle
import math
import os
current_path = os.path.dirname(__file__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    # hacked ...
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# NEW


class MedNLIProcessor(DataProcessor):
    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, "mli_train_v1.jsonl")
        return self._create_examples(file_path)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        file_path = os.path.join(data_dir, "mli_dev_v1.jsonl")
        return self._create_examples(file_path)

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        file_path = os.path.join(data_dir, "mli_test_v1.jsonl")
        return self._create_examples(file_path)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, file_path):
        examples = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                example = json.loads(line)
                examples.append(
                    InputExample(guid=example['pairID'], text_a=example['sentence1'], 
                        text_b=example['sentence2'], label=example['gold_label']))

        return examples

# Ching-Heng ====================
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class CaroditProcessor(DataProcessor):
    def read_variable(self, open_path):
        with open(open_path, 'rb') as file_pi:
            val = pickle.load(file_pi)
            return val

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, "training_bert.pickle")
        return self._create_examples(file_path)

    def get_dev_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, "test_bert.pickle")
        return self._create_examples(file_path)

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        file_path = os.path.join(data_dir, "test_bert.pickle")
        return self._create_examples(file_path)

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
                'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']

    def _create_examples(self, data_dir):
        examples = []
        gid = 0
        data = self.read_variable(data_dir)
        X = data['processed_content']
        Y = data[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
                'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
        for x, y in zip(X, Y):
            guid = "guid-%s" % (gid)
            examples.append(
                InputExample(guid=guid, text_a=x, text_b=None, label=y))
            gid += 1
        return examples


class RestrokeProcessor(DataProcessor):
    def read_variable(self, open_path):
        with open(open_path, 'rb') as file_pi:
            val = pickle.load(file_pi)
            return val

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, "training_bert.pickle")
        return self._create_examples(file_path)

    def get_dev_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, "test_bert.pickle")
        return self._create_examples(file_path)

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        file_path = os.path.join(data_dir, "test_bert.pickle")
        return self._create_examples(file_path)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data_dir):
        examples = []
        gid = 0
        data = self.read_variable(data_dir)
        X = data['processed_content']
        Y = data[['label']].values
        for x, y in zip(X, Y):
            guid = "guid-%s" % (gid)
            examples.append(
                InputExample(guid=guid, text_a=x, text_b=None, label=y[0]))
            gid += 1
        return examples

# =========================


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, task_name):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        if seq_len > max_len:
            max_len = seq_len
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if task_name == 'carotid':
            label_id = []
            for label in example.label:
                label_id.append(float(label))
        else:
            label_id = label_map[example.label]

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if task_name == 'carotid':
                logger.info("label: %s (id = %s)" % (example.label, label_id))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    
    print('Max Sequence Length: %d' %max_len)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def setup_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese, biobert.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--model_loc', type=str, default='', help="Specify the location of the bio or clinical bert model")
    parser.add_argument('--vocab_loc', type=str, default='', help="Specify the location of token vocabulary file is necessary")
    return parser

def main():
# def main(args):
    parser = setup_parser()
    args = parser.parse_args()

    # specifies the path where the biobert or clinical bert model is saved
    if args.bert_model == 'biobert' or args.bert_model == 'clinical_bert' or args.bert_model == 'stroke_bert' or args.bert_model == 'stroke_biobased_bert':
        args.bert_model = args.model_loc

    print(args.bert_model)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "mednli": MedNLIProcessor,
        "carotid": CaroditProcessor,
        "restroke": RestrokeProcessor
    }

    num_labels_task = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "mednli": 3,
        "carotid": 17,
        "restroke": 2
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()


    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(args.vocab_loc, do_lower_case=args.do_lower_case)

    print('TRAIN')
    train = processor.get_train_examples(args.data_dir)
    print([(train[i].text_a,train[i].text_b, train[i].label)  for i in range(3)])
    print('DEV')
    dev = processor.get_dev_examples(args.data_dir)
    print([(dev[i].text_a,dev[i].text_b, dev[i].label) for i in range(3)])
    print('TEST')
    test = processor.get_test_examples(args.data_dir)
    print([(test[i].text_a,test[i].text_b, test[i].label) for i in range(3)])


  
    train_examples = None
    num_train_optimization_steps = -1
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    if task_name == 'carotid':
        model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                  cache_dir=cache_dir,
                  num_labels = num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, task_name)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        if task_name == 'carotid':
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        else:
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * WarmupLinearSchedule(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        if task_name == 'carotid':
            model = BertForMultiLabelSequenceClassification(config, num_labels=num_labels)
        else:
            model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        if task_name == 'carotid':
            model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
        else:
            model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, task_name)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        if task_name == 'carotid':
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        else:
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        all_logits = None
        all_labels = None

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
 
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            if task_name == 'carotid':
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

                if all_labels is None:
                    all_labels = label_ids.detach().cpu().numpy()
                else:
                    all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
            else:
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

        if task_name == 'carotid':
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(num_labels):
                fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            save_path = os.path.join(args.output_dir, "eval_prediction.pickle")
            predic_result={'all_logits': all_logits, 'all_labels': all_labels}
            with open(save_path, 'wb') as file_pi:
                pickle.dump(predic_result, file_pi)

            result = {'eval_loss': eval_loss,
                      'roc_auc': roc_auc}
        else:
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            test_examples = processor.get_test_examples(args.data_dir)
            test_features = convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, tokenizer, task_name)
            logger.info("***** Running testing *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            if task_name == 'carotid':
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)
            else:
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

            all_logits = None
            all_labels = None

            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
     
            for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_test_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                if task_name == 'carotid':
                    if all_logits is None:
                        all_logits = logits.detach().cpu().numpy()
                    else:
                        all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

                    if all_labels is None:
                        all_labels = label_ids.detach().cpu().numpy()
                    else:
                        all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
                else:
                    logits = logits.detach().cpu().numpy()
                    if all_logits is None:
                        all_logits = logits
                    else:
                        all_logits = np.concatenate((all_logits, logits), axis=0)

                    label_ids = label_ids.to('cpu').numpy()
                    if all_labels is None:
                        all_labels = label_ids
                    else:
                        all_labels = np.concatenate((all_labels, label_ids), axis=0)

                    tmp_test_accuracy = accuracy(logits, label_ids)

                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy

                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            if task_name == 'carotid':
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(num_labels):
                    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                save_path = os.path.join(args.output_dir, "test_prediction.pickle")
                predic_result = {'all_logits': all_logits, 'all_labels': all_labels}
                with open(save_path, 'wb') as file_pi:
                    pickle.dump(predic_result, file_pi)

                result = {'test_loss': test_loss,
                          'roc_auc': roc_auc}
            else:
                test_loss = test_loss / nb_test_steps
                test_accuracy = test_accuracy / nb_test_examples
                loss = tr_loss/nb_tr_steps if args.do_train else None

                save_path = os.path.join(args.output_dir, "test_prediction.pickle")
                predic_result = {'all_logits': all_logits, 'all_labels': all_labels}
                with open(save_path, 'wb') as file_pi:
                    pickle.dump(predic_result, file_pi)

                result = {'test_loss': test_loss,
                          'test_accuracy': test_accuracy,
                          'global_step': global_step,
                          'loss': loss}

            output_test_file = os.path.join(args.output_dir, "test_results.txt")
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

class Hacked_arg:
    def __init__(self, data_dir, bert_model, task_name, output_dir, cache_dir, max_seq_length,
                 do_train, do_eval, do_test, do_lower_case, train_batch_size, eval_batch_size,
                 learning_rate, num_train_epochs, warmup_proportion, no_cuda, local_rank, seed,
                 gradient_accumulation_steps, fp16, loss_scale, server_ip, server_port, model_loc):
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.task_name = task_name
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.do_lower_case = do_lower_case
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.server_ip = server_ip
        self.server_port = server_port
        self.model_loc = model_loc


if __name__ == "__main__":
    # python run_classifier.py --bert_model clinical_bert --model_loc ../models/strokeBERT_dis_100000 --data_dir ./data --task_name carotid --output_dir ./c_output --max_seq_length 400 --train_batch_size 24 --do_train  --do_test
    main()

    #  directly run the code
    # print(current_path)
    # hacked_arg = Hacked_arg(
    #     data_dir=os.path.join(current_path, 'data'),
    #     bert_model='stroke_bert',
    #     # bert_model='bert-base-cased',
    #     # bert_model='clinical_bert',
    #     task_name='carotid',
    #     output_dir=os.path.join(current_path, 'output'),
    #     cache_dir='',
    #     max_seq_length=400,
    #     do_train=True,
    #     do_eval=False,
    #     do_test=True,
    #     do_lower_case=False,
    #     train_batch_size=24,
    #     eval_batch_size=24,
    #     learning_rate=5e-5,
    #     num_train_epochs=10.0,
    #     warmup_proportion=0.1,
    #     no_cuda=False,
    #     local_rank=-1,
    #     seed=369,
    #     gradient_accumulation_steps=1,
    #     fp16=False,
    #     loss_scale=0,
    #     server_ip='',
    #     server_port='',
    #     model_loc=os.path.join('..', 'models', 'strokeBERT_dis_100000')
    # )
    # main(hacked_arg)
