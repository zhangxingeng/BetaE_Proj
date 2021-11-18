import os
import torch
import pickle
import json
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from betae.model import KGReasoning
from betae.dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from betae.query import query_name_dict
from betae.util import flatten_query, eval_tuple
from betae.logging import log_metrics

def generate_tensorboard_files(args):
    ''' Tensorboard Support '''
    if not args.do_train: # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    return writer

def read_dataset_info(args):
    ''' Read node and edge number '''
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    return nentity, nrelation

def load_data(args, dataset, tasks, all_tasks, name_query_dict, logger):
    ''' Load queries and remove queries not in tasks '''
    logger.info("loading data")
    dataset.train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    dataset.train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    dataset.valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    dataset.valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    dataset.valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    dataset.test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    dataset.test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    dataset.test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in dataset.train_queries:
                del dataset.train_queries[query_structure]
            if query_structure in dataset.valid_queries:
                del dataset.valid_queries[query_structure]
            if query_structure in dataset.test_queries:
                del dataset.test_queries[query_structure]

def split_train_queries_path_and_other(args, dataset):
    ''' split queries into no branching and branching groups '''
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


def create_valid_dataloader(args, dataset, logger):
    ''' Create Dataloader for validation queries '''
    for query_structure in dataset.valid_queries:
        logger.info(query_name_dict[query_structure]+": "+str(len(dataset.valid_queries[query_structure])))
    ''' Convert queries into DataLoader '''
    valid_queries = flatten_query(dataset.valid_queries)
    valid_dataloader = DataLoader(
        TestDataset(valid_queries, args.nentity, args.nrelation,), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn)
    return valid_dataloader

def create_test_dataloader(args,dataset, logger):
    ''' Create Dataloader for test queries '''
    for query_structure in dataset.test_queries:
        logger.info(query_name_dict[query_structure]+": "+str(len(dataset.test_queries[query_structure])))
    ''' create test DataLoader '''
    test_queries = flatten_query(dataset.test_queries)
    test_dataloader = DataLoader(
        TestDataset(test_queries, args.nentity, args.nrelation,), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn)
    return test_dataloader

def create_model(args, logger):
    ''' Creating Beta Embedding model '''
    model = KGReasoning(
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        beta_mode = eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict
    )

    ''' Recording parameters '''
    logger.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logger.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logger.info('Parameter Number: %d' % num_params)
    if args.cuda: model = model.cuda()

    return model


def load_model(args, model, logger):
    ''' Create training optimizer '''
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate )
        warm_up_steps = args.max_steps // 2
    else: current_learning_rate, optimizer, warm_up_steps = None, None, None

    ''' load checkpoint if specified '''
    if args.checkpoint_path is not None:
        logger.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logger.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0
    return optimizer, init_step, warm_up_steps, current_learning_rate

def save_model(model, optimizer, save_variable_list, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    ''' Evaluate queries in dataloader '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1
    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics