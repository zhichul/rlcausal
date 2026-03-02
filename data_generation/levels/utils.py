import itertools
import numpy as np
import networkx as nx

from ..variable_elimination_with_heuristic import min_fill_variable_elimination

def cpt_to_strs(cpt: np.ndarray, parents: list[str], values: list[list[str]], rounding=2):
    lines = []
    for value_inds in itertools.product(*[list(range(len(val))) for val in values[:-1]]): # the last one is child values, which we vectorize when printing
        assignment_tuple = [(parents[i], values[i][value_ind]) for i, value_ind in enumerate(value_inds)]
        if len(value_inds) == 0:
            lines.append(f"P({parents[-1]}) = {str(cpt[value_inds].round(rounding).tolist())}")
        else:
            lines.append(f"P({parents[-1]} | {','.join([f'{parent}={val}'for parent, val in assignment_tuple])}) = {str(cpt[value_inds].round(rounding).tolist())}")
    return lines

def values_to_str(g, varnames):
    value_lines = []
    for varname, nval in sorted(zip(varnames, g.n_vals)):
        value_lines.append(f'{varname} can take values in {list(range(nval))}')
    value_str = "\n".join(value_lines)
    return value_str

def parametrization_to_str(g, varnames):
    cpt_lines = []
    for cpt_i, cpt in enumerate(g.cpts):
        cpt_lines.append(f"CPTs for {varnames[cpt_i]}:")
        parents = g.parents[cpt_i] + [cpt_i]
        parent_names = [varnames[pa] for pa in parents]
        parent_values = [[str(v) for v in range(g.n_vals[pa])] for pa in parents] # default value is just 0...n_vals-1
        cpt_lines.extend(cpt_to_strs(cpt, parent_names, parent_values))
        cpt_lines.append("")
    cpt_str = '\n'.join(cpt_lines)
    return cpt_str

def remove_incoming_edges(g, src):
    g = g.copy()
    g.remove_edges_from(list(g.in_edges(src)))
    return g

def relevant_and_dependent_subgraph(g: nx.DiGraph, query: int, obs: int = None):
    q_set = {query}
    o_set = {obs} if obs is not None else set()
    q_ans = nx.ancestors(g, query)
    o_ans = nx.ancestors(g, obs) if obs is not None else set()
    sub: nx.DiGraph = g.subgraph(q_ans | o_ans | q_set | o_set).copy()
    dependent = set()
    for node in sub.nodes():
        if not nx.d_separated(sub, {node}, q_set, o_set):
            dependent.add(node)
    subsub: nx.DiGraph = sub.subgraph(dependent | q_set | o_set).copy()
    return subsub

def relevant_subgraph(g: nx.DiGraph, query: int, obs: int = None):
    """
    Returns the subgraph of g induced by ancestors of query and obs. 
    When query happen to equal obs then the subgraph only has a single node.
    """
    q_set = {query}
    o_set = {obs} if obs is not None else set()
    if query == obs:
        sub = g.subgraph(q_set).copy()
    else:
        q_ans = nx.ancestors(g, query)
        o_ans = nx.ancestors(g, obs) if obs is not None else set()
        sub: nx.DiGraph = g.subgraph(q_ans | o_ans | q_set | o_set).copy()
    return sub

def get_baseline(g, gnx, k):
    rel_subgraph = relevant_subgraph(gnx, k)
    nodes = set(rel_subgraph.nodes())
    vars_to_eliminate =  nodes - {k}
    res, pars, n1, n2 =  min_fill_variable_elimination([cpt for i, cpt in enumerate(g.cpts) if i in nodes], [iparent for i, iparent in enumerate(g.iparents) if i in nodes], vars_to_eliminate, renormalize=True)
    return res

if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_nodes_from([0,1,2,3,4])
    g.add_edges_from([(1,3), (2,3), (3,0), (2,0), (4, 2)])
    print(g.nodes())
    print(g.edges())
    r = relevant_and_dependent_subgraph(g, 3, 2)
    print(r.nodes())
    print(r.edges())