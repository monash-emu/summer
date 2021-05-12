from unittest import mock

from typing import Callable, Optional
import random

import igraph as ig
import networkx as nx
from networkx import drawing
import matplotlib.pyplot as plt


from .model import AgentModel

NodeFunc = Callable[[AgentModel, int], str]


def plot_model_networks(
    model: AgentModel, seed: int = 1, get_node_color: Optional[NodeFunc] = None, draw_kwargs={}
):
    """
    Plots model networks onto a figure.
    Returns a matplotlib figure object.

    Mappings can be use to generate per-node kwargs.
    Kwargs can also be passed directly to draw_networkx.
    https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
    """
    # Combine all network graphs into one big graph.
    graphs = [model.networks.vals["graph"][i] for i in model.networks.query.ids()]
    graph = nx.compose_all(graphs)

    # Convert networkx graph into igraph format, use that to make a layout.
    random.seed(seed)
    igraph_graph = ig.Graph.from_networkx(graph)
    layout = igraph_graph.layout_fruchterman_reingold()

    # Plot actual graph
    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=120)
    kwargs = {"ax": ax, "node_size": 50}

    if get_node_color:
        colors = []
        for node_id in graph.nodes.keys():
            rgb_color_tuple = get_node_color(model, node_id)
            rgb_color_tuple_normalised = tuple([n / 255.0 if n > 1 else n for n in rgb_color_tuple])
            colors.append(rgb_color_tuple_normalised)

        kwargs["node_color"] = colors

    with_labels = draw_kwargs.pop("with_labels", True)
    arrows = draw_kwargs.pop("arrows", True)
    kwargs = {**kwargs, **draw_kwargs}

    # Draw the network to the axis.
    # Copied from NetworkX because we can't override their plt.draw_if_interactive() code
    # inside of drawing.draw_networkx
    if any([k not in VALID_KWDS for k in kwargs]):
        invalid_args = ", ".join([k for k in kwargs if k not in VALID_KWDS])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwargs.items() if k in VALID_NODE_KWDS}
    edge_kwds = {k: v for k, v in kwargs.items() if k in VALID_EDGE_KWDS}
    label_kwds = {k: v for k, v in kwargs.items() if k in VALID_LABEL_KWDS}

    drawing.draw_networkx_nodes(graph, layout, **node_kwds)
    drawing.draw_networkx_edges(graph, layout, arrows=arrows, **edge_kwds)
    if with_labels:
        drawing.draw_networkx_labels(graph, layout, **label_kwds)

    return fig


# Copied from NetworkX because we can't override their plt.draw_if_interactive() code
# inside of drawing.draw_networkx
VALID_NODE_KWDS = (
    "nodelist",
    "node_size",
    "node_color",
    "node_shape",
    "alpha",
    "cmap",
    "vmin",
    "vmax",
    "ax",
    "linewidths",
    "edgecolors",
    "label",
)

VALID_EDGE_KWDS = (
    "edgelist",
    "width",
    "edge_color",
    "style",
    "alpha",
    "arrowstyle",
    "arrowsize",
    "edge_cmap",
    "edge_vmin",
    "edge_vmax",
    "ax",
    "label",
    "node_size",
    "nodelist",
    "node_shape",
    "connectionstyle",
    "min_source_margin",
    "min_target_margin",
)

VALID_LABEL_KWDS = (
    "labels",
    "font_size",
    "font_color",
    "font_family",
    "font_weight",
    "alpha",
    "bbox",
    "ax",
    "horizontalalignment",
    "verticalalignment",
)

VALID_KWDS = VALID_NODE_KWDS + VALID_EDGE_KWDS + VALID_LABEL_KWDS
