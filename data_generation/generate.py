from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import tqdm
import yaml
import addict
import networkx as nx


from .levels import interventional, observational, counterfactual
from .utils import NoDefaultADDict, copy_file, write_data
from .dag import sample_dag

def generate_examples(config: addict.Dict, uniqueness_check=True):
    """
    Input: config file
    
    Config specifies n_models unique models (SCM)'s per train/dev/test split, and n_questions_per_model for each unique SCM.
    The set of SCM's used in train/dev/test are disjoint.

    1. We first sample unique DAGs and split into train/dev/test.
    2. For each DAG, we sample SCM parameters, and then sample n_questions_per_model questions, and solve for the answer.

    Writes:
        1. Parquet files for each of train/dev/test split.
        2. Raw dumps of the SCM.
    """
    np.random.seed(config.seed)
    if not config.model.type == "discrete_cpt":
        raise ValueError(f"Unknown model type: {config.model.type}. Did you have typo?")

    if config.type == 'interventional':
        data_generator = interventional
    elif config.type == 'observational':
        data_generator = observational
    elif config.type == 'counterfactual':
        data_generator = counterfactual
    else:
        raise ValueError(f"Unknown generator type: {config.type}. Did you have a typo?")

    # Step 1, sample DAGs
    splits = ['train', 'dev', 'test']
    per_subsplit_model_counts = {'train': config.data.n_models.train, 
                    'dev': config.data.n_models.dev,
                    'test': config.data.n_models.test,}
    per_subsplit_quest_counts = {'train': config.data.n_questions_per_model.train, 
                    'dev': config.data.n_questions_per_model.dev,
                    'test': config.data.n_questions_per_model.test,}

    total_model_counts = defaultdict(int)
    for split in splits:
        n_vars_sweep = config.model.n_vars.train if split == 'train' else config.model.n_vars.eval
        for n_vars in n_vars_sweep:
            total_model_counts[n_vars] += per_subsplit_model_counts[split]

    dags = defaultdict(list)
    for n_vars in tqdm.tqdm(total_model_counts, desc='Generating graphs'):
        print(f'Generating ({n_vars}) count = {total_model_counts[n_vars]}')
        seen = set()
        for i in tqdm.tqdm(range(total_model_counts[n_vars])):
            tries = 0
            while True:
                g = sample_dag(n_vars)
                tries += 1
                if tries > config.data.max_tries_for_uniqueness:
                    raise RuntimeError(f"Tried {tries} times without getting unique graph for evaluation")
                graph_hash = nx.weisfeiler_lehman_graph_hash(g.nx_graph) # guaranteed non-isomorphic graphs will get different hashes
                if not uniqueness_check or graph_hash not in seen:
                    seen.add(graph_hash)
                    break
            dags[n_vars].append((i, g))

    for val in dags.values():
        np.random.shuffle(val)

    # Step 2, sample SCM parameters, and a batch of questions and answers for each SCM
    print('per_subsplit_quest_counts', per_subsplit_quest_counts)
    for split in tqdm.tqdm(splits, desc="split"):
        n_vars_sweep = config.model.n_vars.train if split == 'train' else config.model.n_vars.eval
        n_vals_sweep = config.model.n_vals.train if split == 'train' else config.model.n_vals.eval
            
        for n_vars in n_vars_sweep:
            split_group_dags = dags[n_vars]
            dags[n_vars] = split_group_dags[per_subsplit_model_counts[split]:]
            split_graphs = split_group_dags[:per_subsplit_model_counts[split]]
            for n_vals in n_vals_sweep:
                subsplit = f'{split}_n{n_vars}v{n_vals}'
                split_data = []
                for i, dag in tqdm.tqdm(split_graphs):
                    ex, g, _ = data_generator.generate_example_group(config, 
                        n_quest=per_subsplit_quest_counts[split], 
                        n_vars=n_vars, 
                        n_vals=n_vals, 
                        dag=dag, 
                        extra={            
                        'group_id': f'{i}n{n_vars}',
                        'dag_id': f'{i}n{n_vars}v{n_vals}',
                        'split': split,
                        'subsplit': subsplit, # files will be saved by subsplits
                        }) # this id is only local to this generation to distinguish groups
                    split_data.append((g, ex))
                write_data(config, split_data, subsplit, format='json' if 'format' not in config.data else config.data.format)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', type=Path
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config = NoDefaultADDict(config)
    
    os.makedirs(config.out_dir, exist_ok=True)
    copy_file(args.config, os.path.join(config.out_dir, args.config.name))
    generate_examples(config)