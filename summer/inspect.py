"""Tools for probing, querying, inspecting and drawing CompartmentalModels

The main entry point for this is the ModelProbe class
"""

import re
from typing import Iterable, Set

import networkx as nx

from summer import CompartmentalModel
from summer.compartment import Compartment
from summer.flows import BaseFlow


def query_compartments(m: CompartmentalModel, query: dict):
    if "name" in query:
        query = query.copy()
        name = query.pop("name")
        return m.get_matching_compartments(name, query)
    else:
        _strata = frozenset(query.items())
        return [c for c in m.compartments if c._has_strata(_strata)]


def query_flows(
    m: CompartmentalModel, flow_name: str = None, source: dict = None, dest: dict = None
):
    if flow_name is not None:
        re_pat = re.compile(flow_name)
        flows = [f for f in m._flows if re_pat.match(f.name)]
    else:
        flows = m._flows

    if source:
        if "name" in source:
            source = source.copy()
            name = source.pop("name")
            flows = [f for f in flows if f.source and f.source.name == name]
        else:
            source = frozenset(source.items())
            flows = [f for f in flows if f.source and f.source._has_strata(source)]

    if dest:
        if "name" in dest:
            dest = dest.copy()
            name = dest.pop("name")
            flows = [f for f in flows if f.dest and f.dest.name == name]
        else:
            dest = frozenset(dest.items())
            flows = [f for f in flows if f.dest and f.dest._has_strata(source)]

    return flows


def flows_to_compartments(m: CompartmentalModel, flows: Iterable[BaseFlow]):
    comps = []
    for f in flows:
        if f.source:
            comps.append(f.source)
        if f.dest:
            comps.append(f.dest)
    return set(comps)


def build_compartment_flow_map(m: CompartmentalModel):
    out_map = {c: set() for c in m.compartments}
    for f in m._flows:
        if f.source:
            out_map[f.source].add(f)
        if f.dest:
            out_map[f.dest].add(f)
    return out_map


class ModelProbe:
    def __init__(self, model: CompartmentalModel):
        self.model = model
        self._compartment_flow_map = build_compartment_flow_map(model)

    def compartments_to_flows(self, compartments: Iterable[Compartment]) -> Set[BaseFlow]:
        flows = []
        for c in compartments:
            flows += self._compartment_flow_map[c]
        return set(flows)

    def flows_to_compartments(self, flows: Iterable[BaseFlow]) -> Set[Compartment]:
        return flows_to_compartments(self.model, flows)

    def query_compartments(self, query: dict = None):
        query = query or {}
        return query_compartments(self.model, query)

    def query_flows(self, flow_name: str = None, source: dict = None, dest: dict = None):
        return query_flows(self.model, flow_name, source, dest)

    def get_model_subset(self, comp_query: dict = None, flow_query: dict = None):

        comp_query = comp_query or {}
        flow_query = flow_query or {}

        comps = self.query_compartments(comp_query)
        flows = self.query_flows(**flow_query)

        matched_comps = self.flows_to_compartments(flows)
        matched_flows = self.compartments_to_flows(comps)

        return matched_comps.intersection(set(comps)), matched_flows.intersection(set(flows))

    def draw_flow_graph(self, comp_query: dict = None, flow_query: dict = None):

        # FIXME: Not yet working - need to update with new computegraph methods

        raise NotImplementedError()
        # comp_query = comp_query or {}
        # flow_query = flow_query or {}

        # comps, flows = self.get_model_subset(comp_query, flow_query)
        # digraph = model_to_digraph(comps, flows)

        # pos = nx.nx_agraph.graphviz_layout(digraph)

        # edge_trace, node_trace = ngraph.get_traces(digraph, pos)

        # node_trace.marker.color = ['#ff3333' if 'infectious' in str(node) else \
        # '#dd44dd' if 'latent' in str(node) else '#22dd22' for node in digraph.nodes]
        # node_text = [
        #   get_node_text_summer(probe.model, n, node) for n,node in digraph.nodes.items()
        # ]
        # node_trace.text = node_text

        # title = 'Compartmental Model'

        # return get_graph_figure(edge_trace, node_trace, title)


"""
Tools specific to handling models as networkx graph structures
"""


def model_to_digraph(compartments: Iterable[Compartment], flows: Iterable[BaseFlow]):
    g = nx.DiGraph()
    for c in compartments:
        g.add_node(c)

    def is_fully_mapped(f, comps):
        return f.source in comps and f.dest in comps

    for f in flows:
        if is_fully_mapped(f, compartments):
            g.add_edge(str(f.source), str(f.dest))

    return g
