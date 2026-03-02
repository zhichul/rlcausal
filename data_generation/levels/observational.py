from __future__ import annotations
import copy
import addict
import numpy as np
from ..causal_graph import DiscreteSCM, Sample, sample_causal_graph
from ..variable_elimination_with_heuristic import min_fill_variable_elimination
import numpy as np
from .utils import get_baseline, parametrization_to_str, relevant_subgraph, values_to_str

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
        2.1 Sample an observational query.
        2.2 Compute reference solution for the query.
        2.3 Verbalize the question into a prompt.

    Outputs:
        examples: a list of dict, each dict is a question
        g: the SCM
        obs_samples: the samples used to construct each question
    """
    assert config.data.n_observe == 1
    assert config.data.prompt.reference.type == "distribution"

    # sample a graph
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
    obs_samples = []
    examples = []
    
    for i in range(n_quest):
        # sample conditions
        observed = np.random.choice(varinds, size=(config.data.n_observe,), replace=False).tolist()
        question_vars = varinds

        # setup the observation for var elim
        obs, obs_sample, cpts, parents = setup_observation(g, observed, sample.endos[i], sample.exos[i])
        observation = ', '.join([f'{varnames[j]}={obs_val}' for j, obs_val in obs])
        obs_samples.append(obs_sample)

        # sample a question var
        k = np.random.choice(question_vars)
        rel_subgraph = relevant_subgraph(gnx, k, obs=observed[0]) # assumes single observation

        # reference for sample mode
        reference_distribution, num_prods, num_sums = get_reference_distribution(g=g, gnx=gnx, observed=observed, obs_sample=obs_sample, k=k, cpts=cpts, parents=parents)
        reference_distribution = str(reference_distribution.round(2).tolist())

        # get baseline value
        baseline = str(get_baseline(g, gnx, k).round(2).tolist())
        tvd_no_obs = (np.abs(np.array(eval(baseline)) - np.array(eval(reference_distribution)))).sum() / 2
        tvd_no_intv = 0

        question = f'Question: What is the marginal distribution of {varnames[k]} given it is observed that {observation}?'
        reference = reference_distribution

        # finish the example
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
                'observed': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'intervention': [
                ],
                'question': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in question_vars if j != k
                ],
            },
            'sample_id': i,
            'min_fill_products': num_prods,
            'min_fill_sums': num_sums,
            'ancestor_size': rel_subgraph.number_of_nodes() if k not in observed else 1,
            'tvd_no_obs': tvd_no_obs,
            'tvd_no_intv': tvd_no_intv,
            'is_core': tvd_no_obs > 0,
        }
        ex.update(extra)
        examples.append(ex)
    return examples, g, obs_samples


def setup_observation(g: DiscreteSCM, observed: list[int], sample_endo: np.ndarray, sample_exo: np.ndarray):
    obs = []
    cpts = copy.copy(g.cpts)
    parents = copy.copy(g.iparents)
    for j in observed:
        obs_val = sample_endo[j].item()
        obs.append((j, obs_val))
        
        # prep for variable eliminatino
        obs_factor = np.zeros(g.n_vals[j])
        obs_factor[obs_val] = 1.
        cpts.append(obs_factor)
        parents.append([j]) # add a 0-dim new variable to the end that depends on the value of the observed, it'll not be summed out during variable elimination

    # reuse sample for conditional sample (view as rejection sampling)
    obs_sample = Sample(sample_endo, sample_exo, dict(), dict(), dict(), dict())
    return obs, obs_sample, cpts, parents

def get_reference_distribution(*, g, gnx, observed, obs_sample, k, cpts, parents):
    rel_subgraph = relevant_subgraph(gnx, k, obs=observed[0]) # assumes single observation
    
    # reference for distribution and argmax modes
    if k not in observed:
        vars_to_eliminate = [varind for varind in rel_subgraph.nodes() if varind != k] # sum everyone out
        cpts = [g.cpts[varind] for varind in rel_subgraph.nodes()] + [cpts[-1]]
        parents = [g.iparents[varind] for varind in rel_subgraph.nodes()] + [parents[-1]]
        reference_distribution, _, num_prods, num_sums = min_fill_variable_elimination(cpts, parents, vars_to_eliminate, renormalize=True)
    else:
        num_prods = num_sums = 0
        reference_distribution = np.zeros(g.n_vals[k])
        reference_distribution[obs_sample.endos[k].item()] = 1.
    return reference_distribution, num_prods, num_sums
    