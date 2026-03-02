"""
Variable elimination with min-fill heuristic. 
"""
from functools import reduce
import numpy as np


import logging
logger = logging.getLogger()
from itertools import combinations
from collections import defaultdict

LETTERS = [chr(x) for x in (list(range(65, 91)) + list(range(97, 123)))]

def ascii(i: int):
    return LETTERS[i]

def eliminate_one(cpts: list[np.ndarray], parents: list[list[int]], var: int):
    logger.debug(f"#factors={len(cpts)}")
    operands = []
    in_indices = []
    out_indices = dict()
    eliminated = []
    kept = []
    var_dim = None

    # create the summation index and operands
    for i in range(len(cpts)):
        pas = parents[i]
        cpt = cpts[i]
        if var in pas:
            eliminated.append(i)
            operands.append(cpt)
            in_indices.append("".join([ascii(pa) for pa in pas]))
            for j, pa in enumerate(pas):
                if pa != var:
                    out_indices[ascii(pa)] = pa
                else:
                    if var_dim is None:
                        var_dim = cpt.shape[j]
                    else:
                        assert var_dim == cpt.shape[j], (var, var_dim, cpt.shape[j], pas, [cpt.shape for cpt in cpts])

        else:
            kept.append(i)
    
    assert len(eliminated) > 0, (parents, var)
    # einsum
    subscript = f'{",".join(in_indices)}->{"".join(list(out_indices.keys()))}'
    new_factor = np.einsum(subscript, *operands, optimize=True)
    new_pas = list(out_indices.values())

    # each cell sums over var_dim products of size len(eliminated)
    num_sums = reduce(int.__mul__, new_factor.shape if len(new_pas) > 0 else [1]) * (var_dim - 1) 
    num_products = reduce(int.__mul__, new_factor.shape if len(new_pas) > 0 else [1]) * (len(eliminated) - 1)

    logger.debug(subscript)
    logger.debug([operand.shape for operand in operands] + [new_factor.shape])
    logger.debug(f"-{len(eliminated)} factors of sizes {[len(parents[i]) for i in eliminated]}")
    
    # construct constant, new factors, and parents
    parents = [parents[i] for i in kept]
    cpts = [cpts[i] for i in kept]
    if len(new_pas) == 0:
        # summed out the entire factor into a scaler
        constant = new_factor
        logger.debug(f"produced a constant")
    else:
        constant = 1
        cpts.append(new_factor)
        parents.append(list(new_pas))
        logger.debug(f"+1 factor of size [{len(new_pas)}]")
    return constant, cpts, parents, num_sums, num_products

def build_interaction_graph(factors: list[list[int]]):
    graph = defaultdict(set)
    for factor in factors:
        for u, v in combinations(factor, 2):
            graph[u].add(v)
            graph[v].add(u)
    return graph

def count_fill_in_fast(graph: dict[int, set[int]], var: int):
    neighbors = graph[var]
    
    # Total potential edges in a clique of size k

    k = len(neighbors)
    total_possible = k * (k - 1) // 2

    # Existing edges among neighbors
    existing_edges = 0
    for u in neighbors:
        existing_edges += len(graph[u].intersection(neighbors))
    existing_edges //= 2  # since counted twice

    return total_possible - existing_edges

def min_fill_ordering(factors: list[list[int]], vars_to_eliminate: set[int] = None):
    graph = build_interaction_graph(factors)
    elimination_order = []

    # eliminate everything if not supplied
    if vars_to_eliminate is None:
        vars_to_eliminate = set(graph.keys())
    else:
        if not isinstance(vars_to_eliminate, set):
            vars_to_eliminate = set(vars_to_eliminate)

    while vars_to_eliminate:
        fill_counts = {
            var: count_fill_in_fast(graph, var) for var in vars_to_eliminate
        }
        var_to_eliminate = min(fill_counts, key=fill_counts.get)
        elimination_order.append(var_to_eliminate)

        # Add fill-in edges
        neighbor_set = set(graph[var_to_eliminate])
        for u in neighbor_set:
            missing = neighbor_set - graph[u] - {u}
            for v in missing:
                graph[u].add(v)
                graph[v].add(u)

        # Remove the variable
        for neighbor in graph[var_to_eliminate]:
            graph[neighbor].remove(var_to_eliminate)
        del graph[var_to_eliminate]
        vars_to_eliminate.remove(var_to_eliminate)

    return elimination_order

def min_fill_variable_elimination(cpts: list[np.ndarray], parents: list[list[int]], vars_to_eliminate: list[int], renormalize=False):
    graph = build_interaction_graph(parents)
    elimination_order = []
    constants = 1
    total_num_sums = 0
    total_num_prods = 0
    # eliminate everything if not supplied
    if vars_to_eliminate is None:
        vars_to_eliminate = set(graph.keys())
    else:
        if not isinstance(vars_to_eliminate, set):
            vars_to_eliminate = set(vars_to_eliminate)

    while vars_to_eliminate:
        fill_counts = {
            var: count_fill_in_fast(graph, var) for var in vars_to_eliminate
        }
        var_to_eliminate = min(fill_counts, key=fill_counts.get)
        elimination_order.append(var_to_eliminate)

        # Add fill-in edges
        neighbor_set = set(graph[var_to_eliminate])
        for u in neighbor_set:
            missing = neighbor_set - graph[u] - {u}
            for v in missing:
                graph[u].add(v)
                graph[v].add(u)

        # Remove the variable
        for neighbor in graph[var_to_eliminate]:
            graph[neighbor].remove(var_to_eliminate)
        del graph[var_to_eliminate]
        vars_to_eliminate.remove(var_to_eliminate)

        constant, cpts, parents, num_sums, num_products = eliminate_one(cpts, parents, var_to_eliminate)
        constants *= constant
        total_num_sums += num_sums
        total_num_prods += num_products
    
    # post-processing
    # make sure all remaining factors are of the same shape, if more than one remaining,
    # this can happen if e.g. there was an original factor (2,3) then (2,3,1) and 1 got eliminated
    dims = [len(cpt.shape) for cpt in cpts]
    assert len(set(dims)) == 1
    shape = [cpt.shape for cpt in cpts]
    assert len(set(shape)) == 1

    # this is when there's more than one factor remaining, we compile them into one factor
    if len(cpts) > 1:
        ret = constants
        for cpt in cpts:
            ret = ret * cpt
        result = [ret, parents[0], total_num_prods, total_num_sums]
    else:
        result = [cpts[0] * constants, parents[0], total_num_prods, total_num_sums]
    if renormalize:
        result[0] = result[0] / result[0].sum()
    return result
        
def variable_elimination(cpts: list[np.ndarray], parents: list[list[int]], order: list[int], renormalize=False):
    parents = {i: {pa: None for pa in pas} for i, pas in enumerate(parents)}
    cpts = {i: cpt for i, cpt in enumerate(cpts)}
    next_id = len(parents)
    constants = 1
    logger.debug(f"#factors={len(cpts)}")
    for var in order:
        operands = []
        in_indices = []
        out_indices = dict()
        eliminated = []
        for i in cpts:
            pas = parents[i]
            cpt = cpts[i]
            if var in pas:
                eliminated.append(i)
                operands.append(cpt)
                in_indices.append("".join([ascii(pa) for pa in pas]))
                for pa in pas:
                    if pa != var:
                        out_indices[ascii(pa)] = pa
        subscript = f'{",".join(in_indices)}->{"".join(list(out_indices.keys()))}'
        logger.debug(subscript)
        new_factor = np.einsum(subscript, *operands, optimize=True)
        logger.debug([operand.shape for operand in operands] + [new_factor.shape])

        new_pas = list(out_indices.values())

        logger.debug(f"-{len(eliminated)} factors of sizes {[len(parents[i]) for i in eliminated]}")
        for i in eliminated:
            del parents[i]
            del cpts[i]

        if len(new_pas) == 0:
            # summed out the entire factor into a scaler
            constants = constants * new_factor
            logger.debug(f"produced a constant")
        else:
            cpts[next_id] = new_factor
            parents[next_id] = {pa: None for pa in new_pas}
            next_id += 1
            logger.debug(f"+1 factor of size [{len(new_pas)}]")
        logger.debug(f"#factors={len(cpts)}")

    dims = [len(cpt.shape) for cpt in cpts.values()]
    assert len(set(dims)) == 1
    shape = [cpt.shape for cpt in cpts.values()]
    assert len(set(shape)) == 1
    if len(cpts) > 1:
        ret = constants
        for cpt in cpts.values():
            ret = ret * cpt
        result = [ret, list(next(iter(parents.values())).keys())]
    else:
        result = [next(iter(cpts.values())) * constants, list(next(iter(parents.values())).keys())]
    if renormalize:
        result[0] = result[0] / result[0].sum()
    return result

if __name__ == "__main__":
    from .causal_graph import sample_causal_graph

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')    
    def demo():
        sem = sample_causal_graph(seed=3, n_vals=12)
        print("model 1", sem)
        marginal, parents = variable_elimination([sem.cpts[0], sem.cpts[3]], [sem.iparents[0], sem.iparents[3]], [0])
        manual_marginal = np.matmul(sem.cpts[0], sem.cpts[3])
        np.testing.assert_allclose(marginal, manual_marginal)
        print("test 1 done, leaf to root, two nodes")
        print('\n##############\n')

        marginal, parents = variable_elimination([sem.cpts[0], sem.cpts[3], sem.cpts[2], sem.cpts[4]], [sem.iparents[0], sem.iparents[3], sem.iparents[2], sem.iparents[4]], [4, 2, 0])
        np.testing.assert_allclose(marginal, manual_marginal)
        print("test 2 done, added extra distractor nodes that don't matter")
        print('\n##############\n')

        # this is a graph with two diamonds
        sem = sample_causal_graph(seed=15, n_vals=12)
        print("model 2", sem)
        marginal3, parents3 = variable_elimination([sem.cpts[0], sem.cpts[1], sem.cpts[2], sem.cpts[3]], [sem.iparents[0], sem.iparents[1], sem.iparents[2], sem.iparents[3]], [0, 1, 2])
        manual_joint_12 = (sem.cpts[0][:, None, None] * sem.cpts[1][:,:,None] * sem.cpts[2][:, None, :]).sum(0)
        manual_marginal_3 = np.matmul(manual_joint_12.reshape(-1), sem.cpts[3].reshape(-1, sem.cpts[3].shape[-1]))
        np.testing.assert_allclose(marginal3, manual_marginal_3)
        print("test 3 done, diamond")
        print('\n##############\n')

        marginal3, parents3 = variable_elimination([sem.cpts[0], sem.cpts[1], sem.cpts[2], sem.cpts[3]], [sem.iparents[0], sem.iparents[1], sem.iparents[2], sem.iparents[3]], [2, 1, 0])
        np.testing.assert_allclose(marginal3, manual_marginal_3)
        print("test 4 done, diamond but change order")
        print("here's the min-fill order")
        print(min_fill_ordering([sem.iparents[0], sem.iparents[3], sem.iparents[2], sem.iparents[4]], [2,1,0]))

        print('\n##############\n')
        marginal3, parents3, tp, ts = min_fill_variable_elimination([sem.cpts[0], sem.cpts[1], sem.cpts[2], sem.cpts[3]], [sem.iparents[0], sem.iparents[1], sem.iparents[2], sem.iparents[3]], [1,2,0])
        np.testing.assert_allclose(marginal3, manual_marginal_3)
        print(f"total_products={tp}, total_sums={ts}")

        print('\n##############\n')

        # some more tests
        print("here are some constants test")
        marginal4, parents4, tp, ts = min_fill_variable_elimination([np.array([[[1,1],[1,1]],[[1,1],[1,1]]]), np.array([[1,1],[1,1]]), np.array([1,1])], [[0,1,2], [0,1], [3]], [2, 3])
        print(marginal4, parents4)
        print(f"total_products={tp}, total_sums={ts}")

    demo()