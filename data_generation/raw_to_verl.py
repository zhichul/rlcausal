import json
import random
import os
import os.path
from datasets import load_dataset, concatenate_datasets, Dataset

import argparse

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/ppo/n10v2')
    parser.add_argument('--data_source', default=os.path.join(os.environ['ROOT'], 'data/2025-04-06/interventional_distribution'))
    parser.add_argument('--data_subsplits', nargs="+", default=['n10v2'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--monitor_count', type=int, default=None)
    parser.add_argument('--first', type=int, default=None)
    parser.add_argument('--filter_key', type=str, default=None)
    parser.add_argument('--filter_val', type=str, default=None, help='use json format')
    parser.add_argument('--filter_fn', type=str, default=None, help='use eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--difficulty_field', type=str, default='ancestor_depth')
    parser.add_argument('--donot_name_by_difficulty', action='store_true')
    parser.add_argument('--donot_keep_original_fields', action='store_true')
    parser.add_argument('--filter_before', action='store_true', default=None, help='use json format')

    args = parser.parse_args()
    random.seed(args.seed)

    train_datasets = []
    dev_datasets = []
    test_datasets = []
    for subsplit in args.data_subsplits:
        train_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'train_{subsplit}.parquet'))['train'])
        dev_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'dev_{subsplit}.parquet'))['train'])
        test_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'test_{subsplit}.parquet'))['train'])
        
    train_dataset = concatenate_datasets(train_datasets)
    dev_dataset = concatenate_datasets(dev_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    if args.filter_key is not None and args.filter_before:
        if args.filter_val is not None:
            assert args.filter_fn is None
            filter_val = json.loads(args.filter_val)
            train_dataset=  train_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
            dev_dataset=  dev_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
            test_dataset=  test_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
        elif args.filter_fn is not None:
            assert args.filter_val is None
            filter_fn = eval(args.filter_fn) # dangerous
            train_dataset=  train_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))
            dev_dataset=  dev_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))
            test_dataset=  test_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))

    if args.debug:
        train_dataset = Dataset.from_dict(train_dataset[:100])
        dev_dataset = Dataset.from_dict(dev_dataset[:100])
        test_dataset = Dataset.from_dict(test_dataset[:100])
    if args.first:
        train_dataset = Dataset.from_dict(train_dataset[:args.first])
        dev_dataset = Dataset.from_dict(dev_dataset[:args.first])
        test_dataset = Dataset.from_dict(test_dataset[:args.first])

    if args.monitor_count:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        dev_dataset = dev_dataset.shuffle(seed=args.seed)
        test_dataset = test_dataset.shuffle(seed=args.seed)
        train_dataset = Dataset.from_dict(train_dataset[:args.monitor_count])
        dev_dataset = Dataset.from_dict(dev_dataset[:args.monitor_count])
        test_dataset = Dataset.from_dict(test_dataset[:args.monitor_count])

    if args.filter_key is not None and not args.filter_before:
        if args.filter_val is not None:
            assert args.filter_fn is None
            filter_val = json.loads(args.filter_val)
            train_dataset=  train_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
            dev_dataset=  dev_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
            test_dataset=  test_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
        elif args.filter_fn is not None:
            assert args.filter_val is None
            filter_fn = eval(args.filter_val) # dangerous
            train_dataset=  train_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))
            dev_dataset=  dev_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))
            test_dataset=  test_dataset.filter(lambda ex: filter_fn(ex[args.filter_key]))

    def make_map_fn(split):

        def process_fn(example, idx):
            prompt_raw = example.pop('example')
            system_prompt = prompt_raw['system']
            user_prompt = prompt_raw['user']
            answer = prompt_raw['reference']
            subsplit = example['subsplit']
            group_id = example['group_id']
            dag_id = example['dag_id']
            data = {
                "data_source": (args.data_source if args.data_name is None else args.data_name) + (f"_d{example[args.difficulty_field]}" if not args.donot_name_by_difficulty else ""),
                "prompt": [{
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.loads(answer)
                },
                "extra_info": {
                    'split': split,
                    'subsplit': subsplit,
                    'group_id': group_id,
                    'dag_id': dag_id,
                    'index': idx,
                    "prompt":system_prompt + user_prompt,
                    "answer": answer
                }
            }
            return data

        return process_fn
    remove_columns = []
    if args.donot_keep_original_fields:
        remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=remove_columns)
    dev_dataset = dev_dataset.map(function=make_map_fn('dev'), with_indices=True, remove_columns=remove_columns)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=remove_columns)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, 'dev.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))