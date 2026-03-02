from __future__ import annotations
import copy
from functools import reduce
import itertools
import addict
import numpy as np
from ..variable_elimination_with_heuristic import min_fill_variable_elimination
from .observational import get_reference_distribution as grdist_obs, setup_observation
from .interventional import get_reference_distribution as grdist_intv, setup_intervention
from .utils import remove_incoming_edges, parametrization_to_str, values_to_str
import numpy as np
from .utils import get_baseline, relevant_subgraph
import networkx as nx

from ..causal_graph import DiscreteSCM, sample_causal_graph

def generate_example_group(config: addict.Dict, n_quest: int, n_vars: int, n_vals: int|list[int], dag=None, extra={}):
    """
    Input: 
        config:
        n_quest: number of questions to generate
        n_vars: number of variables
        n_vals: cardinalities of the variables
        dag: optionally an dag structure to follow
        extra: additional key/values to put into the example as meta data
    
    Steps
    1. Sample a SCM parametrization. Create a copy to build the twin with. See https://arxiv.org/abs/1206.5294 for example.
    2. Draw n_quest samples.
    2. For each question:
        2.1 Sample a counterfactual query.
        2.2 Compute reference solution for the query.
        2.3 Verbalize the question into a prompt.

    Outputs:
        examples: a list of dict, each dict is a question
        g: the SCM
        intv_samples: the samples used to construct each question
    """
    assert config.data.n_observe == 1 and config.data.n_intervene == 1
    assert config.data.prompt.reference.type == "distribution"

    # sample a graph
    g = sample_causal_graph(n_vars=n_vars, n_vals=n_vals, dag=dag)
    gnx = g.nx_graph
    g_twin = to_twin_graph(g)
    gnx_twin = g_twin.nx_graph
    
    # build shared verbalizations
    # sample a random rename of the variables
    varinds = list(range(n_vars))
    varnames = np.array([f'v{i}' for i in varinds])
    varnames_str = ', '.join(varnames)
    np.random.shuffle(varnames) # reindex

    # verbalize graph, value of variables, and parametrization
    g_str = g.to_dot(varnames)
    value_str = values_to_str(g, varnames)
    cpt_str = parametrization_to_str(g, varnames)
    
    # start building examples
    system_prompt = config.data.prompt.system
    sample = g.draw(n_quest)
    intv_samples = []
    examples = []

    for i in range(n_quest):

        # sample a observation and intervention var each, observation should be downstream of intervention var
        observed, intervened = draw_condition_and_intervention(gnx=gnx, n_observe=config.data.n_observe,
                                                               n_intervene=config.data.n_intervene,
                                                               n_vars=n_vars)
        question_vars = varinds

        # setup the observation
        observation = observation_to_str(observed, sample.endos[i], varnames)
        
        # setup the intervention
        intv, intv_sample, twin_cpts, twin_parents = setup_counterfactual(g=g, g_twin=g_twin, observed=observed, intervened=intervened, sample_endo=sample.endos[i], sample_exo=sample.exos[i])
        intv_samples.append(intv_sample)

        # sample a question var
        k = np.random.choice(question_vars)
        twin_k = 3 * k + 2

        # reference for all modes
        assert len(intervened) == 1
        # compute difficulty
        gmod = remove_incoming_edges(gnx_twin, 3 * intervened[0] + 2)
        rel_subgraph = relevant_subgraph(gmod, twin_k, obs=3 * observed[0] + 1)

        # compute reference distribution
        vars_to_eliminate = [var for var in rel_subgraph.nodes() if var != twin_k] # do not sum out k, in case we intervened on k and asked about k...
        twin_cpts = [twin_cpts[varind] for varind in rel_subgraph.nodes()] + [twin_cpts[-1]]
        twin_parents = [twin_parents[varind] for varind in rel_subgraph.nodes()] + [twin_parents[-1]]


        reference_distribution, _, num_prods, num_sums = min_fill_variable_elimination(twin_cpts, twin_parents, vars_to_eliminate, renormalize=True) 
        _, obs_sample, obs_cpts, obs_parents = setup_observation(g, observed, sample.endos[i], sample.exos[i])
        _, _, intv_cpts, intv_parents = setup_intervention(g, intervened, sample.exos[i], draw=False, intv=intv)
        baseline_no_intv, _, _ = grdist_obs(g=g, gnx=gnx, observed=observed, obs_sample=obs_sample, k=k, cpts=obs_cpts, parents=obs_parents)
        baseline_no_obs, _, _ = grdist_intv(g=g, gnx=gnx, intervened=intervened, k=k, cpts=intv_cpts, parents=intv_parents)
        baseline_no_both = get_baseline(g=g, gnx=gnx, k=k)
        baseline_no_intv = str(baseline_no_intv.round(2).tolist())
        baseline_no_obs = str(baseline_no_obs.round(2).tolist())
        baseline_no_both = str(baseline_no_both.round(2).tolist())

        reference_distribution = str(reference_distribution.round(2).tolist())
        tvd_no_obs = (np.abs(np.array(eval(baseline_no_obs)) - np.array(eval(reference_distribution)))).sum() / 2
        tvd_no_intv = (np.abs(np.array(eval(baseline_no_intv)) - np.array(eval(reference_distribution)))).sum() / 2
        tvd_no_both = (np.abs(np.array(eval(baseline_no_both)) - np.array(eval(reference_distribution)))).sum() / 2


        question = f'Question: What is the marginal distribution of {varnames[k]} given we first observed {observation} and then intervened to set {varnames[intv[0][0]]} to {intv[0][1]}?'
        reference = reference_distribution
    
        # finish the example
        format_dict = dict(size=g.n_vars,
                        values=value_str,
                        var_names=varnames_str,
                        question=question,
                        cpts=cpt_str,
                        graph=g_str)
        if config.data.n_observe > 0:
            format_dict['observation'] = observation

        user_prompt = config.data.prompt.user.format(**format_dict)

        ex = {
            'example': {
                'system': system_prompt,
                'user': user_prompt,
                'reference': reference,
                'baseline_no_obs': baseline_no_obs,
                'baseline_no_intv': baseline_no_intv,
                'baseline_no_both': baseline_no_both
            },
            'meta': {
                'variable_names': varnames.tolist(),
                'n_vars': g.n_vars,
                'n_vals': g.n_vals,
                'graph_description': g.to_str(varnames.tolist()), 
            },
            'formal_example': {
                'observed': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'intervention': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in intervened
                ],
                'question': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in question_vars if j != k
                ],
            },
            'sample_id': i,
            'min_fill_products': num_prods,
            'min_fill_sums': num_sums,
            'ancestor_size': rel_subgraph.number_of_nodes() if k not in intervened else 1,
            'tvd_no_obs': tvd_no_obs,
            'tvd_no_intv': tvd_no_intv,
            'tvd_no_both': tvd_no_both,
            'is_core': tvd_no_both > 0 and tvd_no_intv > 0 and tvd_no_obs > 0,
            'is_semicore': tvd_no_both > 0 or tvd_no_intv > 0 or tvd_no_obs > 0,
        }
        ex.update(extra)
        examples.append(ex)
    
    return examples, g, intv_samples

def draw_condition_and_intervention(*, gnx: nx.Digraph, n_observe: int, n_intervene: int, n_vars: int):
    """
    Randomly choose intervention vars, then observation vars **downstream** of intervention vars.
    This produces a counterfactual graph.
    Assuming variable indices are topologically sorted, i.e. lower index means upstream.
    """
    varinds = list(range(n_vars))
    intervened_range = varinds[0: n_vars-1] 
    intervened_shuffle = np.random.choice(intervened_range, size=(n_intervene,), replace=False).tolist()
    intervened = intervened_shuffle[:n_intervene]
    intervened_downstream = set()
    for intv_var in intervened:
        intervened_downstream |= nx.descendants(gnx, intv_var) | {intv_var}
    observed_shuffle = np.random.choice(sorted(intervened_downstream), size=(1,), replace=False).tolist()
    if n_observe > 1:
        observed_shuffle += np.random.choice([v for v in varinds if v != observed_shuffle[0]], size=(n_observe-1,), replace=False).tolist()
    observed = observed_shuffle[:n_observe]
    return observed, intervened

def observation_to_str(observed: list[int], sample: np.ndarray, varnames: list[str]):
    obs = []
    for j in observed:
        obs_val = sample[j].item()
        obs_str = f'{varnames[j]} = {obs_val}'
        obs.append(obs_str)
    observation = ', '.join(obs)
    return observation

def to_selector_cpt(cpt: np.ndarray):
    par_shape = cpt.shape[:-1]
    selector_size = cpt.shape[-1] ** reduce(int.__mul__, par_shape, 1)
    par_size = reduce(int.__mul__, par_shape, 1)

    offset = np.broadcast_to(np.arange(par_size).reshape(par_shape + (1,)), par_shape + (selector_size,))
    new_cpt = np.broadcast_to(np.arange(selector_size).reshape([1] * len(par_shape) + [selector_size]), par_shape + (selector_size,))
    new_cpt = (new_cpt //  cpt.shape[-1] ** (par_size-1-offset)) % cpt.shape[-1]
    new_cpt = np.eye(cpt.shape[-1])[new_cpt] # turn one-hot
    return new_cpt

def outer(vecs):
    names = [chr(ord('a') + i) for i in range(len(vecs))]
    left = ','.join(names)
    right = ''.join(names)
    return np.einsum(f'{left}->{right}', *vecs)

def to_twin_graph(g: DiscreteSCM):
    n_vars = g.n_vars * 3 # layout is [v1 selector, v1, v1 twin, ..., vn selector, vn, vn twin]
    n_vals = list(itertools.chain([g.n_vals[i] ** reduce(int.__mul__, cpt.shape[:-1], 1), g.n_vals[i], g.n_vals[i]] for i, cpt in enumerate(g.cpts)))
    parents = [[] for _ in range(n_vars)]
    for i, pas in enumerate(g.parents):
        parents[3*i + 1] = [3*pa + 1 for pa in pas] + [3*i] # original var
        parents[3*i + 2] = [3*pa + 2 for pa in pas] + [3*i] # twin var
    n_indep = sum(len(pa) == 0 for pa in parents)
    cpts = [None] * n_vars
    for i, cpt in enumerate(g.cpts):
        assert cpt.flags['C_CONTIGUOUS'] # assume rowwise layout
        cpts[3*i] = outer(list(cpt.reshape(-1, cpt.shape[-1]))).reshape(-1)
        cpts[3*i + 1] = to_selector_cpt(cpt)
        cpts[3*i + 2] = to_selector_cpt(cpt)
    return DiscreteSCM(n_vars, n_vals, n_indep, parents, cpts)

def setup_counterfactual(g: DiscreteSCM, g_twin: DiscreteSCM, observed: list[int], intervened: list[int], sample_endo: np.ndarray, sample_exo: np.ndarray):
    clamp_endo = {}
    clamp_exo = {vi: val for vi, val in enumerate(sample_exo)}
    intv = []

    cpts = copy.copy(g_twin.cpts)
    parents = copy.copy(g_twin.iparents)
    for j in intervened:
        # Again, following Lampinen et al., 2023 
        # Passive learning of active causal strategies in agents and language models
        intv_val = np.random.choice(g.n_vals[j]) 
        intv.append((j, intv_val))
        clamp_endo[j] = intv_val

        # update cpts with graph surgery corresponding to intervention (used later in variable elimination)
        parents[3 * j + 2] = [3 * j + 2]
        cpts[3 * j + 2] = np.zeros(g.n_vals[j])
        cpts[3 * j + 2][intv_val] = 1.

    for j in observed:
        # prep for variable eliminatino
        obs_val = sample_endo[j].item()
        obs_factor = np.zeros(g.n_vals[j])
        obs_factor[obs_val] = 1.
        cpts.append(obs_factor)
        parents.append([3 * j + 1]) # add a 0-dim new variable to the end that depends on the value of the observed, it'll not be summed out during variable elimination

    intv_sample = g.draw(clamp_endo=clamp_endo, clamp_exo=clamp_exo) # not actually sampling since exo is fully clamped
    return intv, intv_sample, cpts, parents

if __name__ == '__main__':
    print("here's some examples of the selector mechanism and the twin network graph")
    cpt = np.arange(8).reshape((2,2,2))
    cpt = cpt / np.sum(cpt, axis=-1, keepdims=True)
    print(to_selector_cpt(cpt))
    g = sample_causal_graph(n_vars=3, n_vals=2, dag=None)
    g_twin = to_twin_graph(g)
    print(g_twin)