from __future__ import annotations
import copy
import addict
import numpy as np
from ..variable_elimination_with_heuristic import min_fill_variable_elimination
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
    1. Sample a SCM parametrization.
    2. Draw n_quest samples.
    2. For each question:
        2.1 Sample an interventional query.
        2.2 Perform graph surgery based on intervention.
        2.3 Compute reference solution for the query.
        2.4 Verbalize the question into a prompt.

    Outputs:
        examples: a list of dict, each dict is a question
        g: the SCM
        intv_samples: the samples used to construct each question
    """
    assert config.data.n_observe == 0 and config.data.n_intervene == 1
    assert config.data.prompt.reference.type == "distribution"

    # sample a SCM
    g = sample_causal_graph(n_vars=n_vars, n_vals=n_vals, dag=dag)
    gnx = g.nx_graph
    
    # build shared verbalizations
    ## sample a random rename of the variables
    varinds = list(range(n_vars))
    varnames = np.array([f'v{i}' for i in varinds])
    varnames_str = ', '.join(varnames)
    np.random.shuffle(varnames) # reindex

    ## verbalize graph, value of variables, and parametrization
    g_str = g.to_dot(varnames)
    value_str = values_to_str(g, varnames)
    cpt_str = parametrization_to_str(g, varnames)
    
    # start building examples
    system_prompt = config.data.prompt.system
    sample = g.draw(n_quest)
    intv_samples = []
    examples = []
    
    for i in range(n_quest):

        # sample (optional) observations and intervention, currently only intervention
        _, intervened = draw_condition_and_intervention(n_observe=config.data.n_observe,
                                                               n_intervene=config.data.n_intervene,
                                                               n_vars=n_vars)
        
        question_vars = varinds

        # setup the intervention for var elim
        intv, intv_sample, cpts, parents = setup_intervention(g, intervened, sample.exos[i])
        
        intv_samples.append(intv_sample)

        # sample a question var
        k = np.random.choice(question_vars)

        # reference for sample mode
        gmod = remove_incoming_edges(gnx, intervened[0])
        rel_subgraph = relevant_subgraph(gmod, k)
        reference_distribution, num_prods, num_sums = get_reference_distribution(g=g, gnx=gnx, intervened=intervened, k=k, cpts=cpts, parents=parents)
        reference_distribution = str(reference_distribution.round(2).tolist())
    
        question = f'Question: What is the marginal distribution of {varnames[k]} given we intervented to set {varnames[intv[0][0]]} to {intv[0][1]}?'
        reference = reference_distribution

        baseline = str(get_baseline(g, gnx, k).round(2).tolist())
        tvd_no_intv = (np.abs(np.array(eval(baseline)) - np.array(eval(reference_distribution)))).sum() / 2
        tvd_no_obs = 0

        # finish interpolating the prompts
        format_dict = dict(size=g.n_vars,
                        values=value_str,
                        var_names=varnames_str,
                        question=question,
                        cpts=cpt_str,
                        graph=g_str)

        user_prompt = config.data.prompt.user.format(**format_dict)

        ex = {
            'example': {
                'system': system_prompt,
                'user': user_prompt,
                'reference': reference,
                'baseline': baseline,
            },
            'meta': {
                'variable_names': varnames.tolist(),
                'n_vars': g.n_vars,
                'n_vals': g.n_vals,
                'graph_description': g.to_str(varnames.tolist()), 
            },
            'formal_example': {
                'observed': [],
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
            'is_core': tvd_no_intv > 0,
        }
        ex.update(extra)
        examples.append(ex)
    
    return examples, g, intv_samples

def draw_condition_and_intervention(*, n_observe: int, n_intervene: int, n_vars: int):
    """
    Reserving at least the first n_observe variables to not be interevened. 
    Sample intervention vars. Then sample observation vars, restricted to those **upstream** of the intervention.
    Assumes variable indices are topologically sorted so that lower indices are upstream of higher indices.
    """
    varinds = list(range(n_vars))
    intervened_range = varinds[n_observe: n_vars-1]
    intervened_shuffle = np.random.choice(intervened_range, size=(n_intervene,), replace=False).tolist()
    intervened = intervened_shuffle[:n_intervene]
    intervened_min, intervened_max = min(intervened), max(intervened)
    observed_shuffle = np.random.choice(varinds[:intervened_min], size=(n_observe,), replace=False).tolist()
    observed = observed_shuffle[:n_observe]
    return observed, intervened

def setup_intervention(g: DiscreteSCM, intervened: list[int], sample_exo: np.ndarray, draw=True, intv=None):
    clamp_endo = {}
    clamp_exo = {vi: val for vi, val in enumerate(sample_exo)}
    if intv is None:
        intv = [(j, np.random.choice(g.n_vals[j])) for j in intervened]

    cpts = copy.copy(g.cpts)
    parents = copy.copy(g.iparents)
    for idx, j in enumerate(intervened):
        # Again, following Lampinen et al., 2023 
        # Passive learning of active causal strategies in agents and language models
        intv_val = intv[idx][1]
        clamp_endo[j] = intv_val

        # update cpts with graph surgery corresponding to intervention (used later in variable elimination)
        parents[j] = [j]
        cpts[j] = np.zeros(g.n_vals[j])
        cpts[j][intv_val] = 1.

    if draw:
        intv_sample = g.draw(clamp_endo=clamp_endo, clamp_exo=clamp_exo) # not actually sampling since exo is fully clamped
    else:
        intv_sample = None
    return intv, intv_sample, cpts, parents

def get_reference_distribution(*, g, gnx, intervened, k, cpts, parents):
    # reference for distribution and argmax modes
    assert len(intervened) == 1
    gmod = remove_incoming_edges(gnx, intervened[0])
    rel_subgraph = relevant_subgraph(gmod, k)
    vars_to_eliminate = sorted((nx.ancestors(rel_subgraph, k) | set(intervened)) - {k}) # do not sum out k, in case we intervened on k and asked about k...
    cpts = [cpts[i] for i in vars_to_eliminate + [k]]
    parents = [parents[i] for i in vars_to_eliminate + [k]]
    reference_distribution, _, num_prods, num_sums = min_fill_variable_elimination(cpts, parents, vars_to_eliminate)
    return reference_distribution, num_prods, num_sums
