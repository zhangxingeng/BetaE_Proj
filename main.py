import sys, os
from argparse import Namespace
import torch
def setup():
    sys.path.append(r'./')
    if not torch.device('cuda:0'):  raise IOError("Cannot find GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

''' Import helper function'''
from betae.util import parse_time, set_global_seed
from betae.logging import log_load_data, generate_save_path, log_training_info, \
    log_metrics, set_logger
from betae.details import generate_tensorboard_files, read_dataset_info, \
    load_data, split_train_queries_path_and_other, create_valid_dataloader, \
    create_test_dataloader, create_model, load_model, save_model, evaluate

''' Import data '''
from betae.config import config
from betae.query import query_name_dict, name_query_dict, all_tasks

def main():
    args = Namespace(**config)
    dataset = Namespace(**{})
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    args.cur_time = parse_time()

    generate_save_path(args)
    writer = generate_tensorboard_files(args)
    logger = set_logger(args)
    args.nentity, args.nrelation = read_dataset_info(args)

    ''' Loading Data from file '''
    log_load_data(args, logger)
    load_data(args, dataset, tasks, all_tasks, name_query_dict, logger)


    ''' Save training related info '''
    logger.info("Training info:")
    if args.do_train:
        for query_structure in dataset.train_queries:
            logger.info(query_name_dict[query_structure]+": "+str(len(dataset.train_queries[query_structure])))
        ''' Split query into path queries and other queries iterator '''
        train_path_iterator, train_other_iterator = split_train_queries_path_and_other(args, dataset)

    ''' Save validation related info '''
    logger.info("Validation info:")
    if args.do_valid:
        valid_dataloader = create_valid_dataloader(args, dataset, logger)

    ''' Save test related info '''
    logger.info("Test info:")
    if args.do_test: 
        test_dataloader = create_test_dataloader(args, dataset, logger)

    model = create_model(args, logger)
    optimizer, init_step, warm_up_steps, current_learning_rate = load_model(args, model, logger)
    step = init_step 
    log_training_info(args, init_step, current_learning_rate, logger)

    ''' Actual Training '''
    if args.do_train:
        training_logs = []

        for step in range(init_step, args.max_steps): # Training Loop
            ''' Train Path queries '''
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4
            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log: # traverse dict by key
                writer.add_scalar('path_'+metric, log[metric], step)
            ''' Train Non-path queries '''
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)
            training_logs.append(log)

            if step >= warm_up_steps: # update lr every 1.5x previous warm_up_steps 
                current_learning_rate = current_learning_rate / 5
                logger.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_learning_rate)
                warm_up_steps = warm_up_steps * 1.5
            
            if step % args.save_checkpoint_steps == 0: # save checkpoint & model
                save_variable_list = { 'step': step, 'current_learning_rate': current_learning_rate, 'warm_up_steps': warm_up_steps}
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logger.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, dataset.valid_easy_answers, dataset.valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer)

                if args.do_test:
                    logger.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, dataset.test_easy_answers, dataset.test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics, logger)
                training_logs = []

        save_variable_list = {'step': step,  'current_learning_rate': current_learning_rate, 'warm_up_steps': warm_up_steps}
        save_model(model, optimizer, save_variable_list, args)
        
    try:
        print (step)
    except:
        step = 0

    if args.do_test:
        logger.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, dataset.test_easy_answers, dataset.test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)

    logger.info("Training finished!!")

if __name__ == '__main__':
    setup()
    main()