from ast import literal_eval

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


class CausalSystem:
    _project_password = None

    def _sample(self, n_samples):
        raise NotImplementedError

    ##########################################################################################
    # Internal

    # noinspection PyTypeChecker
    def __init__(self):
        self._interventions = None  # type: dict
        self._samples = None  # type: dict
        self._n_samples = None  # type: int

        # Ordering
        self.__ordering = None  # type: list
        self._node_nr = dict()

        # For graph
        self._create_graph = False
        self._current_ancestors = []
        self._ancestors = dict()
        self._descendants = dict()

        # Always ensure a single sample
        _ = self.sample(1)

    def sample(self, n_samples, **interventions):
        # Set
        self._interventions = interventions
        self._samples = dict()
        self.__ordering = []
        self._node_nr = dict()
        self._n_samples = n_samples

        # Compute
        self._create_graph = True
        self._sample(n_samples=n_samples)
        self._create_graph = False

        # Set node-nr
        self._node_nr = {key: nr for nr, key in enumerate(self.__ordering)}

        # Filter keys
        if self._project_password is not None and interventions.get("password", None) == self._project_password:
            index = self.__ordering
        else:
            index = [key for key in self.__ordering if key[0] != "_"]

        # Make table
        table = pd.DataFrame(data=[self._samples[key] for key in index], index=index, dtype=float).T

        # Reset
        self._interventions = None
        self._samples = None
        self._n_samples = None

        # Return
        return table

    def __getitem__(self, item):
        # Remember as ancestor if building graph
        if self._create_graph:
            self._current_ancestors.append(item)

        # Return
        return self._samples[item]

    def __setitem__(self, key, value):
        # Assert new item
        assert isinstance(self._samples, dict) and key not in self._samples

        # Set item
        self.__ordering.append(key)
        self._samples[key] = np.array(value)

        # Intervene if needed
        if isinstance(self._interventions, dict) and key in self._interventions:
            self._samples[key] = np.ones_like(self._samples[key]) * self._interventions[key]

        # Can no longer be changed (we do not allow circular graphs anyway)
        self._samples[key].flags.writeable = False

        # Make graph
        if self._create_graph:
            # Set descendants
            self._descendants[key] = []
            for ancestor in self._current_ancestors:
                self._descendants[ancestor].append(key)

            # Set ancestors
            self._ancestors[key] = self._current_ancestors

            # Reset temporary variables
            self._current_ancestors = []

    @property
    def ancestors(self):
        return self._ancestors

    @property
    def descendants(self):
        return self._descendants

    @property
    def n_nodes(self):
        return len(self.__ordering)

    @property
    def nodes(self):
        return self.__ordering

    @property
    def adjacency_matrix(self):
        graph = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int)
        for ancestor, descendants in self.descendants.items():
            for descendant in descendants:
                graph[self._node_nr[ancestor], self._node_nr[descendant]] = 1

        graph = pd.DataFrame(
            data=graph, index=self.__ordering, columns=self.__ordering
        )

        return graph

    @property
    def edges(self):
        edge_set = set()
        for ancestor, descendants in self.descendants.items():
            for descendant in descendants:
                edge_set.add((ancestor, descendant))
        return edge_set

    def check_correct_graph(self, edge_list):
        # Ensure python object
        for _ in range(3):
            if isinstance(edge_list, str):
                edge_list = literal_eval(edge_list.strip())

        # Format
        edge_set = set([(str(from_node).lower(), str(to_node).lower()) for from_node, to_node in edge_list])

        # Get truth without caring about casing
        true_edge_set = {(from_node.lower(), to_node.lower()) for from_node, to_node in self.edges}

        # Get truth without hidden nodes
        true_edge_set_wo_hidden = {(from_node, to_node) for from_node, to_node in true_edge_set
                                   if "_" not in (from_node[0], to_node[0])}

        # Check
        return edge_set == true_edge_set or edge_set == true_edge_set_wo_hidden

    @property
    def _ordering(self):
        return self.__ordering

    @_ordering.setter
    def _ordering(self, val):
        assert set(val) == set(self.__ordering), f"Ordering must contain all elements of causal graph.\n" \
                                                 f"Graph: {self._ordering}\n" \
                                                 f"New order: {val}\n" \
                                                 f"Difference: {set(val) ^ set(self._ordering)}"
        self.__ordering = val

    def draw_causal_graph(self):
        # Ensure sampled
        _ = self.sample(1)

        # Make graph
        G = nx.DiGraph()

        # Ensure nodes
        G.add_nodes_from(self.nodes)

        # Add edges
        for ancesor, descendants in self.descendants.items():
            G.add_edges_from([(ancesor, val) for val in descendants])

        # Sizes positions and labels for plot
        node_size = 2000
        pos = graphviz_layout(G, prog="dot")
        labels = [val.strip("_") for val in self.nodes]

        # Plot
        plt.close("all")
        plt.title("Causal Graph", fontsize=20)
        nx.draw(
            G, with_labels=True, pos=pos, labels=dict(zip(G.nodes, labels)),
            # arrowstyle=ArrowStyle("simple", head_length=1.3, head_width=1.3, tail_width=.1),
            arrowsize=40,
            width=3,
            node_size=node_size,
            node_color="#ffffff",
            edgecolors="#000000",
            style="solid",
            linewidths=3,
            font_size=20,
        )

        # Fix limits
        offset = np.sqrt(node_size) / 2
        y_lim = min([val for _, val in pos.values()]) - offset, max([val for _, val in pos.values()]) + offset
        x_lim = min([val for val, _ in pos.values()]) - offset, max([val for val, _ in pos.values()]) + offset
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    ####################
    # Pre-made distributions

    def normal(self, mu, std):
        return np.random.randn(self._n_samples) * std + mu

    def categorical(self, probabilities):
        probabilities = np.array(probabilities) / np.sum(probabilities)
        choices = np.array(list(range(len(probabilities))), dtype=np.float)
        return np.random.choice(a=choices, size=self._n_samples, replace=True, p=probabilities)

    def binary(self, p_success):
        return self.categorical(probabilities=np.array([1 - p_success, p_success]))
