# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2022.09.25 - Add support for using quantized Bert model as teacher
#              Meta Platforms, Inc. <zechunliu@fb.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
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
import copy
from kd_learner_glue import KDLearner
from helper import *
from utils_glue import *
from transformer.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from transformer.configuration_bert import BertConfig
from transformer.modeling_bert import BertForSequenceClassification
from transformer.modeling_bert_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", default='tmp', type=str, help='jobid to save training logs')
    parser.add_argument("--data_dir", default=None, type=str,help="The root dir of glue data")
    parser.add_argument("--teacher_model", default='', type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default='', type=str, help="The student model dir.")
    parser.add_argument("--vocab_dir", default='', type=str, help="The vocab.txt dir.")
    parser.add_argument("--task_name", default=None, type=str, help="The name of the glue task to train.")
    parser.add_argument("--output_dir", default='output', type=str,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=None, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", default=None, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd', default=0.01, type=float, metavar='W', help='weight decay')
    parser.add_argument("--num_train_epochs", default=None, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument('--eval_step', type=int, default=100)

    # distillation params
    parser.add_argument('--aug_train', action='store_true',
                        help="Whether using data augmentation or not")
    parser.add_argument('--distill_logit', action='store_true',
                        help="Whether using distillation logits or not")
    parser.add_argument('--distill_rep', action='store_true',
                        help="Whether using distillation reps or not")
    parser.add_argument('--distill_attn', action='store_true',
                        help="Whether using distillation attns or not")
    parser.add_argument('--temperature', type=float, default=1.)

    # quantization params
    parser.add_argument("--weight_bits", default=1, type=int, help="number of bits for weight")
    parser.add_argument("--weight_quant_method", default='bwn', type=str,
                        choices=['bwn', 'uniform'],
                        help="weight quantization methods")
    parser.add_argument("--input_bits",  default=1, type=int,
                        help="number of bits for activation")
    parser.add_argument("--input_quant_method", default='uniform', type=str,
                        help="weight quantization methods")
    parser.add_argument('--not_quantize_attention', action='store_true', help="Keep attention calculations in 32-bit.")

    parser.add_argument('--learnable_scaling', action='store_true', default=True)
    parser.add_argument("--ACT2FN", default='relu', type=str, help='use relu for positive outputs.')

    # training config
    parser.add_argument('--sym_quant_ffn_attn', action='store_true',
                        help='whether use sym quant for attn score and ffn after act') # default asym
    parser.add_argument('--sym_quant_qkvo', action='store_true',  default=True,
                        help='whether use asym quant for Q/K/V and others.') # default sym

    parser.add_argument('--clip_init_file', default='threshold_std.pkl', help='files to restore init clip values.')
    parser.add_argument('--clip_init_val', default=2.5, type=float, help='init value of clip_vals, default to (-2.5, +2.5).')
    parser.add_argument('--clip_lr', default=1e-4, type=float, help='Use a seperate lr for clip_vals / stepsize')
    parser.add_argument('--clip_wd', default=0.0, type=float, help='weight decay for clip_vals / stepsize')

    # layerwise quantization config
    parser.add_argument('--embed_layerwise', default=False, type=lambda x: bool(int(x)))
    parser.add_argument('--weight_layerwise', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--input_layerwise', default=True, type=lambda x: bool(int(x)))

    args = parser.parse_args()
    args.do_lower_case = True

    log_dir = os.path.join(args.output_dir, 'record_%s.log' % args.job_id)
    init_logging(log_dir)

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task_name = args.task_name.lower()

    # restore the default setting if they are None
    if args.learning_rate is None:
        if task_name in default_params:
            args.learning_rate = default_params[task_name]["learning_rate"]

    if args.num_train_epochs is None:
        if task_name in default_params:
            args.num_train_epochs = default_params[task_name]["num_train_epochs"]

    if args.batch_size is None:
        if task_name in default_params:
            args.batch_size = default_params[task_name]["batch_size"]
            #args.batch_size = int(args.batch_size*n_gpu)

    if args.max_seq_length == None:
        if task_name in default_params:
            args.max_seq_length = default_params[task_name]["max_seq_length"]

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    print_args(vars(args))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.teacher_model)
    config.num_labels = num_labels

    student_config = copy.deepcopy(config)
    student_config.weight_bits = args.weight_bits
    student_config.input_bits = args.input_bits
    student_config.weight_quant_method = args.weight_quant_method
    student_config.input_quant_method = args.input_quant_method
    student_config.clip_init_val = args.clip_init_val
    student_config.learnable_scaling = args.learnable_scaling
    student_config.sym_quant_qkvo = args.sym_quant_qkvo
    student_config.sym_quant_ffn_attn = args.sym_quant_ffn_attn
    student_config.embed_layerwise = args.embed_layerwise
    student_config.weight_layerwise = args.weight_layerwise
    student_config.input_layerwise = args.input_layerwise
    student_config.hidden_act = args.ACT2FN
    student_config.not_quantize_attention = args.not_quantize_attention

    num_train_optimization_steps = 0
    if not args.do_eval:
        if args.aug_train:
            train_examples = processor.get_aug_examples(args.data_dir)
        else:
            train_examples = processor.get_train_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.batch_size = args.batch_size // args.gradient_accumulation_steps

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        num_train_optimization_steps = int(
            len(train_features) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    if task_name == "mnli":
        processor = processors["mnli-mm"]()
        if not os.path.exists(args.output_dir + '-MM'):
            os.makedirs(args.output_dir + '-MM')

        mm_eval_examples = processor.get_dev_examples(args.data_dir)
        mm_eval_features = convert_examples_to_features(
            mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)

        logging.info("***** Running mm evaluation *****")
        logging.info("  Num examples = %d", len(mm_eval_examples))

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                     batch_size=args.batch_size)
    else:
        mm_eval_labels = None
        mm_eval_dataloader = None

    if not args.do_eval: # need the teacher model for training
        if hasattr(config, "input_bits") and config.input_bits < 32:
            teacher_model = QuantBertForSequenceClassification.from_pretrained(args.teacher_model, config=config)
        else:
            teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model, config=config)
        teacher_model.to(device)
        if n_gpu > 1:
            teacher_model = torch.nn.DataParallel(teacher_model)
    else:
        teacher_model = None

    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config=student_config)
    student_model.to(device)
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    learner = KDLearner(args, device, student_model, teacher_model,num_train_optimization_steps)

    learner.args.distill_logit = True
    learner.args.distill_rep = True
    learner.args.distill_attn = False

    learner.build(lr=args.learning_rate)
    learner.train(train_examples, task_name, output_mode, eval_labels,
                  num_labels, train_dataloader, eval_dataloader, eval_examples, tokenizer,
                  mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)

    del learner
    return 0


if __name__ == "__main__":
    main()
