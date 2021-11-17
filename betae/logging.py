import logging
import os

def set_logger(args):
    ''' Write logs to console and log file '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig( 
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO, datefmt='%Y%m%d_%H%M%S',
        filename=log_file, filemode='a+')
    logger = logging.getLogger('')
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def log_load_data(args):
    args.logger.info('-------------------------------'*3)
    args.logger.info('Geo: %s' % args.geo)
    args.logger.info('Data Path: %s' % args.data_path)
    args.logger.info('#entity: %d' % args.nentity)
    args.logger.info('#relation: %d' % args.nrelation)
    args.logger.info('#max steps: %d' % args.max_steps)
    args.logger.info('Evaluate unoins using: %s' % args.evaluate_union)

def log_training_info(args, init_step, current_learning_rate):
    if args.geo == 'beta':
        args.logger.info('beta mode = %s' % args.beta_mode)
    args.logger.info('tasks = %s' % args.tasks)
    args.logger.info('init_step = %d' % init_step)
    if args.do_train:
        args.logger.info('Start Training...')
        args.logger.info('learning_rate = %d' % current_learning_rate)
    args.logger.info('batch_size = %d' % args.batch_size)
    args.logger.info('hidden_dim = %d' % args.hidden_dim)
    args.logger.info('gamma = %f' % args.gamma)