import random
import base64
from io import BytesIO
from typing import Callable, Optional, List

import igraph as ig
import networkx as nx
from networkx import drawing
import matplotlib.pyplot as plt
from PIL import Image

from .model import AgentModel

NodeFunc = Callable[[AgentModel, int], str]


def figure_to_buffer(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    return buf


def plots_to_gif_buf(bufs: List[BytesIO], duration: int) -> BytesIO:
    """
    Duration is time per frame, in ms.
    Use Gif Compressor if you want to share:

        https://gifcompressor.com/

    """
    images = [Image.open(buf) for buf in bufs]
    buf = BytesIO()
    images[0].save(
        buf,
        format="gif",
        save_all=True,
        append_images=images,
        optimize=False,
        duration=duration,
        loop=0,
    )
    buf.seek(0)
    return buf


def ipython_display_gif_buf(buf: BytesIO):
    """
    Displays a GIF buffer in Jupyter notebooks.
    """
    from IPython.display import display, HTML

    buf.seek(0)
    gif_base64_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.seek(0)
    img_tag = f"<img src='data:image/gif;base64,{gif_base64_str}'>"
    display(HTML(img_tag))


def ipython_display_plot_buf(buf: BytesIO):
    """
    Displays a plots buffer in Jupyter notebooks.
    """
    from IPython.display import display

    buf.seek(0)
    img = Image.open(buf)
    display(img)


def ipython_display_plot_bufs(bufs: List[BytesIO]):
    """
    Displays an list of plots generated with `ipython_plot_model`
    display in Jupyter notebooks.
    """
    # Imported because not in release version
    import ipywidgets as wg

    def plot_timeslice(i):
        ipython_display_plot_buf(bufs[i])

    wg.interact(plot_timeslice, i=wg.IntSlider(min=0, max=len(bufs) - 1, step=1))


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

    with_labels = draw_kwargs.pop("with_labels", False)
    arrows = draw_kwargs.pop("arrows", True)
    kwargs = {**kwargs, **draw_kwargs}
    drawing.draw_networkx(graph, layout, arrows=arrows, with_labels=with_labels, **kwargs)
    return fig
