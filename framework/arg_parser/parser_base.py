import argparse
parser = argparse.ArgumentParser(description = 'Argument Parser.')

parser.add_argument('--auto_restart', help='auto restore. Enable it on non-stable clusters like philly', default = 1, type = int)
parser.add_argument('--max_num_checkpoint', help='max number of checkpoints', default = 20, type = int)
parser.add_argument('--debug_tag', help='enable debug tag', default = 0, type = int)
#parser.add_argument('--enable_tqdm', help='enable/disable tqdm progbar', default = 1, type = int)

parser.add_argument('--cfg_file', help='config file', default = '', type = str)