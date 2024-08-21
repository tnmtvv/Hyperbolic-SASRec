import argparse
from . import defaults


class ArgparserException(Exception): pass


def parse_args(is_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--time_offset', default=defaults.time_offset_q, type=float)
    parser.add_argument('--target_metric', choices=('HR', 'MRR', 'NDCG'), default=defaults.target_metric, type=str)
    parser.add_argument('--topn', default=10, type=int)
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--exhaustive', default=False, action="store_true")
    parser.add_argument('--grid_steps', default=None, type=int) # 0 means run infinitely, None will use defaults
    parser.add_argument('--check_best', default=False, action="store_true")
    parser.add_argument('--save_config', default=False, action="store_true")
    parser.add_argument('--dump_results', default=False, action="store_true")
    parser.add_argument('--es_tol', default=0.001, type=float)
    parser.add_argument('--es_max_steps', default=2, type=int)
    parser.add_argument('--next_item_only', default=False, action="store_true")
    # saving/resuming studies via RDB:
    parser.add_argument('--study_name', default=None, type=str)
    parser.add_argument('--storage', choices=('sqlite', 'redis'), default=None, type=str)
    args = parser.parse_args()
    validate_args(args, is_test)
    return args


def validate_args(args, is_test):
    if not is_test and not args.config_path:
        # models that require hyper-params tuning must be provided with config file
        # models without hyper-parameters are only valid for test not for tuning phase
        raise ArgparserException('`config_path` must be provided for tuning.')
