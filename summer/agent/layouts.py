"""
NetworX graph visualization layouts
"""
import math
from random import random, seed

import networkx as nx

P = 0.05


def f_a(d, k):
    """Returns the attractive force"""
    return d * d / k


def f_r(d, k):
    """Returns the repulsive force"""
    return 10 * k * k / d


def fruchterman_reingold_layout(
    G: nx.Graph,
    size: int,
    iteration: int = 50,
    layout_seed: int = 1,
):
    """
    Generates a NetworkX compatiable graph layout using the Fructerman + Reingold graph layout algorithm
    https://en.wikipedia.org/wiki/Force-directed_graph_drawing

    Originally posted by  mmisono at
    https://gist.github.com/mmisono/8972731
    """
    length, width = size, size
    seed(layout_seed)

    area = width * length
    k = math.sqrt(area / nx.number_of_nodes(G))

    # initial position
    for node in G.nodes.values():
        node["x"] = width * random()
        node["y"] = length * random()

    t = width / 10
    dt = t / (iteration + 1)

    for _ in range(iteration):
        pos = {}
        for v, node in G.nodes.items():
            pos[v] = [node["x"], node["y"]]

        # calculate repulsive forces
        for v, node in G.nodes.items():
            node["dx"] = 0
            node["dy"] = 0
            for u in G.nodes.keys():
                if v != u:
                    dy = node["y"] - G.nodes[u]["y"]
                    dx = node["x"] - G.nodes[u]["x"]
                    delta = math.sqrt(dx * dx + dy * dy)
                    if delta != 0:
                        d = f_r(delta, k) / delta
                        node["dx"] += dx * d
                        node["dy"] += dy * d

        # calculate attractive forces
        for v, u in G.edges.keys():
            dx = G.nodes[v]["x"] - G.nodes[u]["x"]
            dy = G.nodes[v]["y"] - G.nodes[u]["y"]
            delta = math.sqrt(dx * dx + dy * dy)
            if delta != 0:
                d = f_a(delta, k) / delta
                ddx = dx * d
                ddy = dy * d
                G.nodes[v]["dx"] += -ddx
                G.nodes[u]["dx"] += +ddx
                G.nodes[v]["dy"] += -ddy
                G.nodes[u]["dy"] += +ddy

        # limit the maximum displacement to the temperature t
        # and then prevent from being displace outside frame
        for v in G.nodes.keys():
            dx = G.nodes[v]["dx"]
            dy = G.nodes[v]["dy"]
            disp = math.sqrt(dx * dx + dy * dy)
            if disp != 0:
                d = min(disp, t) / disp
                x = G.nodes[v]["x"] + dx * d
                y = G.nodes[v]["y"] + dy * d
                x = min(width, max(0, x)) - width / 2
                y = min(length, max(0, y)) - length / 2
                G.nodes[v]["x"] = (
                    min(
                        math.sqrt(width * width / 4 - y * y),
                        max(-math.sqrt(width * width / 4 - y * y), x),
                    )
                    + width / 2
                )
                G.nodes[v]["y"] = (
                    min(
                        math.sqrt(length * length / 4 - x * x),
                        max(-math.sqrt(length * length / 4 - x * x), y),
                    )
                    + length / 2
                )

        # cooling
        t -= dt

    pos = {}
    for v in G.nodes.keys():
        pos[v] = [G.nodes[v]["x"], G.nodes[v]["y"]]

    return pos
