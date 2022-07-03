from summer import CompartmentalModel
from summer.compartment import Compartment
import re

# Let's start building reason


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


def flows_to_compartments(m: CompartmentalModel, flows):
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
        self.compartment_flow_map = build_compartment_flow_map(model)

    def compartments_to_flows(self, compartments: list[Compartment]):
        flows = []
        for c in compartments:
            flows += self.compartment_flow_map[c]
        return set(flows)

    def query_compartments(self, query):
        return query_compartments(self.model, query)

    def query_flows(self, flow_name: str = None, source: dict = None, dest: dict = None):
        return query_flows(self.model, flow_name, source, dest)
