import logging

logger = None

def log_message(source, message):
    logger.info('{}:{}'.format(source, message))

def init_logger(log_file):
    global logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    # ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # logger.addHandler(ch)
    logger.addHandler(fh)