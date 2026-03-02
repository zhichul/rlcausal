from __future__ import annotations
import dataclasses
from functools import reduce
import numpy as np
import networkx as nx

from .dag import nx_graph, sample_dag


@dataclasses.dataclass
class DiscreteSCM:
    """
    A discrete SCM with mechanisms specified via conditional probability tables and
    the inverse CDF mechanism `broadcast_invcdf_mechanism`.
    """

    n_vars: int                                         # number of variables
    n_vals: int | list[int]                             # number of values for each variable
    n_indep: int                                        # number of independent variables (i.e. with no parent)             
    parents: list[list | np.ndarray]                    # list of parents of each variable                               
    cpts: list[list | np.ndarray]                       # CPTs of each variable i, of size (nvals[parents[i][0]],...,nvals[self])

    @property
    def iparents(self):
        """
        List of parents of each variable, but includes itself.
        """
        if getattr(self, '_iparents', None) is None:
            self._iparents = [pas + [i] for i, pas in enumerate(self.parents)]
        return self._iparents

    def __eq__(self, value):
        if not (isinstance(value, DiscreteSCM) and 
            value.n_vars == self.n_vars and 
            value.n_vals == self.n_vals and 
            value.n_indep == self.n_indep and 
            deep_list_equal(value.parents, self.parents) and
            deep_list_equal(value.cpts, self.cpts)):
            return False
        return True
            
    def to_str(self, varnames: list[str]):
        lines = []
        for i, (parents, cpt) in enumerate(zip(self.parents, self.cpts)):
            if len(parents) == 0:
                lines.append(f"{varnames[i]} ~ {cpt}")
            else:
                lines.append(f"{varnames[i]} ~ Choice(⋅|{', '.join([varnames[p] for p in parents])})")
        return "\n".join(lines)

    def __str__(self):
        if getattr(self, '_default_varnames', None) is None:
            self._default_varnames = [f'#{i}' for i in range(self.n_vars)]
        return self.to_str(self._default_varnames)

    def to_dot(self, varnames: list[str]) -> str:
        from io import StringIO
        graph = nx.DiGraph()
        raw_inds = list(range(len(varnames)))
        paired_inds = sorted(list(zip(varnames, raw_inds)))
        sorted_varnames = [x[0] for x in paired_inds]
        sorted_raw_inds = [x[1] for x in paired_inds]
        graph.add_nodes_from(sorted_varnames)
        for i in sorted_raw_inds:
            pas = self.parents[i]
            graph.add_edges_from([(varnames[pa], varnames[i]) for pa in pas])
        buffer = StringIO()
        nx.drawing.nx_pydot.write_dot(graph, buffer)
        return buffer.getvalue()

    @property
    def nx_graph(self):
        if getattr(self, '_nx_graph', None) is None:
            self._nx_graph = nx_graph(self.n_vars, self.parents)
        return self._nx_graph

    def draw(self, n: int = None, clamp_endo: dict[int, float|np.ndarray] = None, clamp_exo: dict[int, float|np.ndarray] = None):
        """
        Draw `n` samples, optionally clamping 
            endogenous variables (e.g., v1, v2, v3) for interventions
            exogenous variables (e.g. the noises used to generate v1, v2, v3) for counterfactuals
        """
        if clamp_endo is None:
            clamp_endo = dict()
        if clamp_exo is None:
            clamp_exo = dict()

        # sample exogenous uniform random variables
        exos = np.random.uniform(size=(1, self.n_vars,) if n is None else (n, self.n_vars))
        
        # optionally clamp at `clamp_exo`
        exos_before_clamp = dict()
        for i_exo, exo_val in clamp_exo.items():
            exos_before_clamp[i_exo] = exos[..., i_exo].copy()
            exos[..., i_exo] = exo_val

        # forward sampling endogenous vars, taking clamping into account
        endos = np.zeros((1, self.n_vars,) if n is None else (n, self.n_vars), dtype=np.int64)
        endos_before_clamp = dict()
        for i_endo in range(self.n_vars):
            if len(self.parents[i_endo]) == 0:
                val = broadcast_invcdf_mechanism(self.cpts[i_endo][None, ...], exos[..., i_endo][..., None])
            else:
                val = broadcast_invcdf_mechanism(self.cpts[i_endo][tuple(endos[..., pa] for pa in self.parents[i_endo])], exos[..., i_endo][..., None])
            if i_endo in clamp_endo:
                endos_before_clamp[i_endo] = val
                endos[..., i_endo] = clamp_endo[i_endo]
            else:
                endos[..., i_endo] = val

        if n is None or n == 1:
            endos = endos.squeeze(0)
            exos = exos.squeeze(0)
            endos_before_clamp = broadcast_np_squeeze(endos_before_clamp, 0)
            exos_before_clamp = broadcast_np_squeeze(exos_before_clamp, 0)

        return Sample(endos=endos,
                      exos=exos,
                      clamp_endo=clamp_endo,
                      clamp_exo=clamp_exo,
                      endos_before_clamp=endos_before_clamp,
                      exos_before_clamp=exos_before_clamp)

@dataclasses.dataclass
class Sample:
    """
    Data record of a generated sample from a SCM (optionally) under intervention(s).
    """
    endos: np.ndarray
    exos: np.ndarray
    clamp_endo: dict[int, float|np.ndarray]
    clamp_exo: dict[int, float|np.ndarray]
    endos_before_clamp: dict[int, float|np.ndarray]
    exos_before_clamp: dict[int, float|np.ndarray]

    def __str__(self):
        return (f"endos: {self.endos}\n"
                f"exos: {self.exos}\n"
                f"clamp_endo: {self.clamp_endo}\n"
                f"endos_before_clamp: {self.endos_before_clamp}\n"
                f"clamp_exo: {self.clamp_exo}\n"
                f"exos_before_clamp: {self.exos_before_clamp}\n")

    def __eq__(self, value):
        if not (isinstance(value, Sample) and
                val_equal(value.endos, self.endos) and
                val_equal(value.exos, self.exos) and 
                deep_dict_equal(value.endos_before_clamp, self.endos_before_clamp) and
                deep_dict_equal(value.exos_before_clamp, self.exos_before_clamp) and
                deep_dict_equal(value.clamp_endo, self.clamp_endo) and
                deep_dict_equal(value.clamp_exo, self.clamp_exo)):
            return False
        return True

def broadcast_invcdf_mechanism(p, uni):
    return (p.cumsum(-1) >= uni).argmax(-1) # this relies on argmax returning the first of many ties!

def broadcast_np_squeeze(d: dict[any, np.ndarray], *args, **kwargs):
    return {k: v.squeeze(*args, **kwargs) for k, v in d.items()}

def val_equal(v1, v2):
    match = v1 == v2
    if isinstance(match, bool):
        return match
    elif isinstance(match, np.ndarray):
        return match.all()
    else:
        raise AssertionError

def deep_list_equal(v1, v2):
    assert len(v1) == len(v2)
    for (vv, uu) in zip(v1, v2):
        if not val_equal(vv, uu):
            return False
    return True

def deep_dict_equal(v1, v2):
    assert len(v1) == len(v2) and v1.keys() == v2.keys()
    for k in v1:
        if not val_equal(v1[k], v2[k]):
            return False
    return True

def sample_causal_graph(n_vars: int = 5, n_vals: int | list[int] = 2, seed: int | None = None, dag=None):
    """
    Sample a SCM with `n_vars` variables, each with `n_vals{i}` values, seeded with `seed`.

    This SCM's exogenous variables are U[0,1) and is related to endogenous variables via invcdf on their cpts.
        e.g. for graph v1 -> v2 <- v3
            v1 ~ CPT(v1)
            v3 ~ CPT(v3)
            v2 ~ CPT(v2 | v1, v3)

    Conceptually:
        1. sample a dag, see `sample_dag`.
            1.1 sample the number of independent vars (nodes with indegree 0) m: int ~ Uniform(1, n_vars)
            1.2 randomly sample its number of parents from {1, 2}
            1.3 sample the actual parent(s)
        2. sample cpts row-by-row from i.i.d. flat dirichlet
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(n_vals, int):
        n_vals = [n_vals] * n_vars

    # Step 1, skipped if dag is provided
    if dag is None:
        # if provided don't sample
        dag = sample_dag(n_vars)

    parents = dag.parents
    n_indep = sum(len(pa) == 0 for pa in parents)

    # Step 2
    cpts = [None for _ in range(n_vars)]
    for j in range(n_vars):
        n_val_j = n_vals[j]
        pa_j = parents[j]
        pa_n_val = [n_vals[pa] for pa in pa_j]
        cpt_conds = (reduce(int.__mul__, pa_n_val, 1),)
        cpts[j] = np.random.dirichlet(alpha=[1] * n_val_j, size=cpt_conds).reshape(*(*pa_n_val, n_val_j))

    return DiscreteSCM(
        n_vars=n_vars,
        n_vals=n_vals,
        n_indep=n_indep,
        parents=parents,
        cpts=cpts
    )

if __name__ == "__main__":
    def demo():
        scm = sample_causal_graph(seed=3, n_vals=12, n_vars=5)
        print("Model M1 with 5 variables, and 12 values per variable:")
        print(scm)
        print("----")
        print("1. Draw once from M1:")
        draw = scm.draw()
        print(draw, end="\n----\n")

        print("2. Draw once from M1, locking to the same exogenous values. Endos should match 1.")
        draw_again = scm.draw(clamp_exo=dict(enumerate(draw.exos.T))) # transpose from [sample, var] to [var, sample]
        print(draw_again, end="\n----\n")

        print("3. Draw once, locking to the same exogenous values, and intervening with do(v0)=1. Only v0 and children of v0 should be different from 1's endos.")
        draw_intervene = scm.draw(clamp_endo={0:1}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

        print("\n\nTrying batch interventions.")
        print("4. Draw ten times.")
        draw = scm.draw(n=10)
        print(draw, end="\n----\n")

        print("5. Draw ten times, using same exogenous variables. Endos should match 4.")
        draw_again = scm.draw(n=10, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_again, end="\n----\n")

        print("6. Draw ten times, using same exogenous variables, and intervening with do(v0) = [0,1,...,9]. Only v0 and children of v0 should be different from 4's endos.")
        draw_intervene = scm.draw(n=10, clamp_endo={0:np.arange(10)}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

    demo()