import sys, os
sys.path.append(r'./')
import torch
if not torch.device('cuda:0'):  raise IOError("Cannot find GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

''' Import library '''
from argparse import Namespace

from collections import defaultdict # unlike dict, return default if key not exist
from torch.utils.data import DataLoader
import numpy as np

''' Import helper function and classes'''
from betae.temp import generate_save_path, generate_tensorboard_files, read_dataset_info, \
    split_train_queries_path_and_other, create_valid_dataloader, create_test_dataloader, \
    create_model, load_params_and_optimizer

from betae.logging import log_load_data, log_training_info, set_logger
from betae.util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, save_model, evaluate, log_metrics, load_data
from betae.dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator

''' Import config and queries '''
from betae.config import config
from betae.query import query_name_dict, name_query_dict, all_tasks


args = Namespace(**config)
dataset = Namespace(**{})
param = Namespace(**{})
set_global_seed(args.seed)
tasks = args.tasks.split('.')
args.cur_time = parse_time()

generate_save_path(args)
writer = generate_tensorboard_files(args)
args.logger = set_logger(args)
args.nentity, args.nrelation = read_dataset_info(args)

''' Loading Data from file '''
log_load_data(args)
load_data(args, dataset, tasks, all_tasks, name_query_dict)


''' Save training related info '''
args.logger.info("Training info:")
if args.do_train:
    for query_structure in dataset.train_queries:
        args.logger.info(query_name_dict[query_structure]+": "+str(len(dataset.train_queries[query_structure])))
    ''' Split query into path queries and other queries iterator '''
    train_path_iterator, train_other_iterator = split_train_queries_path_and_other(args, dataset)

''' Save validation related info '''
args.logger.info("Validation info:")
if args.do_valid:
    valid_dataloader = create_valid_dataloader(args, dataset)

''' Save test related info '''
args.logger.info("Test info:")
if args.do_test: 
    test_dataloader = create_test_dataloader(args, dataset)

model = create_model(args)
optimizer, init_step, warm_up_steps, current_learning_rate = load_params_and_optimizer(args, model)
step = init_step 
log_training_info(args, init_step, current_learning_rate)

