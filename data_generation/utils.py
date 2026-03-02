import json
import os
import pickle
import shutil
import string

import addict
import numpy as np
from datasets import Dataset
import pandas as pd

class NoDefaultADDict(addict.Dict):

    def __missing__(self, name):
        raise KeyError(name)

def copy_file(src, tgt):
    with open(src, 'rt') as f:
        with open(tgt, 'wt') as g:
            for line in f:
                g.write(line)
    return None

def random_name(l):
    return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(l))

def write_jsonl(filename, lines, mode='w'):
    with open(filename, mode) as f:
        for e in lines:
            print(json.dumps(e), file=f)

def read_jsonl(filename, first=None):
    out = []
    with open(filename, "rt") as f:
        for i, e in enumerate(f):
            if first is not None and i >= first:
                break
            out.append(json.loads(e))
    return out

def write_parquet(filename, lines):
    dataset = Dataset.from_pandas(pd.DataFrame(data=lines))
    dataset.to_parquet(filename)
    
def write_data(config, data, split, format='json', save_examples_per_group=False):
    if format == 'json':
        file = os.path.join(config.out_dir, f'{split}.jsonl')

        write_jsonl(file, [])
        split_dir = os.path.join(config.out_dir, "by_group", split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        for (g, ex) in data:
            i = ex[0]['group_id']
            # save individual ex and meta data in a folder
            group_dir = os.path.join(split_dir, f'{i}')
            os.makedirs(group_dir, exist_ok=False)
            g_file_i = os.path.join(group_dir, f'graph.pkl')
            with open(g_file_i, "wb") as f:
                pickle.dump(g, f)
            if save_examples_per_group:
                data_file_i = os.path.join(group_dir, f'example.jsonl')
                write_jsonl(data_file_i, ex)
            
            # save to gloabl file
            write_jsonl(file, ex, mode='a')
    if format == 'parquet':
        file = os.path.join(config.out_dir, f'{split}.parquet')
        exes = []
        split_dir = os.path.join(config.out_dir, "by_group", split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        for (g, ex) in data:
            i = ex[0]['group_id']
            # save individual ex and meta data in a folder
            group_dir = os.path.join(split_dir, f'{i}')
            os.makedirs(group_dir, exist_ok=False)
            g_file_i = os.path.join(group_dir, f'graph.pkl')
            with open(g_file_i, "wb") as f:
                pickle.dump(g, f)
            if save_examples_per_group:
                data_file_i = os.path.join(group_dir, f'example.parquet')
                write_parquet(data_file_i, ex)
            exes.extend(ex)
            
            # save to gloabl file
        write_parquet(file, exes)