
import dataclasses
import networkx as nx
import numpy as np

def nx_graph(n_vars, parents):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(n_vars)))
    for i, pas in enumerate(parents):
        graph.add_edges_from([(pa, i) for pa in pas])
    return graph

@dataclasses.dataclass
class DAG:

    n_vars: int
    parents: int

    @property
    def nx_graph(self):
        return nx_graph(self.n_vars, self.parents)

def sample_dag(n_vars):
    """
    Sample a dag:
        1. sample the number of independent vars (nodes with indegree 0) m: int ~ Uniform(1, n_vars)
        2. randomly sample its number of parents from {1, 2}
        3. sample the actual parent(s)
    """
    # graph structure
    n_indep = np.random.randint(1, n_vars)                           # sample num of independent vars
    parents = [[] for _ in range(n_vars)]                            # data structure for parents (factors in SEM)
    for j in range(n_indep, n_vars):                                 # attach later variables sequentially
        n_parent_j = min(j, np.random.randint(1,3))                  # 1 or 2 parents, randint is upper exclusive
        parents[j] = np.random.choice(j, size=n_parent_j,            # sample the parents without replacement
                                      replace=False).tolist()
    return DAG(n_vars, parents)
