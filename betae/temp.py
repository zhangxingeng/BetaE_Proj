import os
import torch
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, dataset
from betae.model import KGReasoning
from betae.dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from betae.query import query_name_dict, name_query_dict, all_tasks
from betae.util import flatten_query, eval_tuple

def generate_save_path(args):
    args.save_path = os.path.join(args.prefix, args.data_path.split('/')[-1], args.tasks.replace('.', '_'), args.geo)
    tmp_str = "g_{}_mode_{}".format(args.gamma, args.beta_mode.replace(',', '_'))
    args.save_path = args.checkpoint_path if args.checkpoint_path != None else os.path.join(args.save_path, tmp_str, args.cur_time)
    if not os.path.exists(args.save_path):
        os.makedirs(os.path.join('./', args.save_path))
    print ("logging to", args.save_path)

def generate_tensorboard_files(args):
    if not args.do_train: # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    return writer



def read_dataset_info(args):
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    return nentity, nrelation



def split_train_queries_path_and_other(args, dataset):
    train_path_queries = defaultdict(set)
    train_other_queries = defaultdict(set)
    path_list = ['1p', '2p', '3p']
    for query_structure in dataset.train_queries:
        if query_name_dict[query_structure] in path_list:
            train_path_queries[query_structure] = dataset.train_queries[query_structure]
        else:
            train_other_queries[query_structure] = dataset.train_queries[query_structure]
    train_path_queries = flatten_query(train_path_queries)
    train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
                                TrainDataset(train_path_queries, args.nentity, args.nrelation, args.negative_sample_size, dataset.train_answers),
                                batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num, collate_fn=TrainDataset.collate_fn ))
    if len(train_other_queries) > 0:
        train_other_queries = flatten_query(train_other_queries)
        train_other_iterator = \
            SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_other_queries, args.nentity, args.nrelation, args.negative_sample_size, dataset.train_answers),
            batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num, collate_fn=TrainDataset.collate_fn ))
    else: train_other_iterator = None
    return train_path_iterator, train_other_iterator


def create_valid_dataloader(args, dataset):
    for query_structure in dataset.valid_queries:
        args.logger.info(query_name_dict[query_structure]+": "+str(len(dataset.valid_queries[query_structure])))
    ''' Convert queries into DataLoader '''
    valid_queries = flatten_query(dataset.valid_queries)
    valid_dataloader = DataLoader(
        TestDataset(valid_queries, args.nentity, args.nrelation,), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn)
    return valid_dataloader

def create_test_dataloader(args,dataset):
    for query_structure in dataset.test_queries:
        args.logger.info(query_name_dict[query_structure]+": "+str(len(dataset.test_queries[query_structure])))
    ''' create test DataLoader '''
    test_queries = flatten_query(dataset.test_queries)
    test_dataloader = DataLoader(
        TestDataset(test_queries, args.nentity, args.nrelation,), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn)
    return test_dataloader

def create_model(args):
    ''' Creating the model '''
    model = KGReasoning(
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda = args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode = eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict
    )

    ''' Recording parameters '''
    args.logger.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        args.logger.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    args.logger.info('Parameter Number: %d' % num_params)
    if args.cuda: model = model.cuda()

    return model


def load_params_and_optimizer(args, model):
    ''' Create training optimizer '''
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate )
        warm_up_steps = args.max_steps // 2

    ''' load checkpoint if specified '''
    if args.checkpoint_path is not None:
        args.logger.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        args.logger.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0
    return optimizer, init_step, warm_up_steps, current_learning_rate

