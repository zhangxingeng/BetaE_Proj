import logging
import os

def set_logger(args):
    global logger
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

def generate_save_path(args):
    ''' generate path to store logs '''
    args.save_path = os.path.join(args.prefix, args.data_path.split('/')[-1], args.tasks.replace('.', '_'), args.geo)
    tmp_str = "g_{}_mode_{}".format(args.gamma, args.beta_mode.replace(',', '_'))
    args.save_path = args.checkpoint_path if args.checkpoint_path != None else os.path.join(args.save_path, tmp_str, args.cur_time)
    if not os.path.exists(args.save_path):
        os.makedirs(os.path.join('./', args.save_path))
    print ("logging to", args.save_path)

def log_metrics(mode, step, metrics, logger):
    ''' Print the evaluation logs '''
    for metric in metrics:
        logger.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def log_load_data(args, logger):
    logger.info('-------------------------------'*3)
    logger.info('Geo: %s' % args.geo)
    logger.info('Data Path: %s' % args.data_path)
    logger.info('#entity: %d' % args.nentity)
    logger.info('#relation: %d' % args.nrelation)
    logger.info('#max steps: %d' % args.max_steps)
    logger.info('Evaluate unoins using: %s' % args.evaluate_union)

def log_training_info(args, init_step, current_learning_rate, logger):
    if args.geo == 'beta':
        logger.info('beta mode = %s' % args.beta_mode)
    logger.info('tasks = %s' % args.tasks)
    logger.info('init_step = %d' % init_step)
    if args.do_train:
        logger.info('Start Training...')
        logger.info('learning_rate = %d' % current_learning_rate)
    logger.info('batch_size = %d' % args.batch_size)
    logger.info('hidden_dim = %d' % args.hidden_dim)
    logger.info('gamma = %f' % args.gamma)