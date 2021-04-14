"""
Script for generating graph for graphical model. The same used in the paper.
This script requires graphviz to run.
"""

import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import argparse


def read_graph(nodes_path, edges_path):
    G = nx.from_dict_of_dicts(json.load(open(edges_path, 'r')))  # type: nx.Graph
    nodes = json.load(open(nodes_path, 'r'))

    for k, v in nodes.items():
        G.add_node(k, **v)

    return G


def plot(graph, savepath=None):
    """
    Draw this individual.
    """

    degrees = {}
    shortest_paths = nx.shortest_path(graph, '0')
    for node in graph.nodes:
        s_path = shortest_paths[node]
        d = 0
        for n in s_path:
            if 'color' not in graph.nodes[n]:
                d += 1
        degrees[node] = d

    max_degree = max(degrees.values())

    colors = list(map(to_hex, viridis(np.linspace(0, 1, num=max_degree * 2))))[(max_degree - 1):]

    for node in graph.nodes:
        if 'color' not in graph.nodes[node]:
            graph.nodes[node]['color'] = colors[degrees[node]]

    fig, ax = plt.subplots(figsize=(16, 10))

    pos = graphviz_layout(graph, root='0', prog='sfdp')

    node_list = graph.nodes(data=True)
    edge_list = graph.edges(data=True)

    node_labels = {node_name: node_attr['label'] for (node_name, node_attr) in node_list}
    node_colors = [node_attr['color'] for (node_name, node_attr) in node_list]
    node_edgecolors = [node_attr['edgecolor'] for (node_name, node_attr) in node_list]

    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=2200, node_color=node_colors, edgecolors=node_edgecolors, alpha=1
    )  # nodes
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edge_list, style='solid', alpha=1)  # edges
    nx.draw_networkx_labels(graph, pos, node_labels, ax=ax, font_size=8)  # node labels

    box = ax.get_position()

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='Value', markerfacecolor='#AAAAAA', markersize=15),
        Line2D([0], [0], marker='o', color='black', label='PBIL', markerfacecolor=colors[0], markersize=15)] + \
    [
        Line2D([0], [0], marker='o', color='black', label='Variable (level %#2.d)' % (i + 1), markersize=15,
               markerfacecolor=color) for i, color in enumerate(colors[1:])
    ]

    ax.legend(handles=legend_elements, loc='lower right', fancybox=True, shadow=True, ncol=1)

    plt.axis('off')

    if savepath is not None:
        plt.savefig(savepath, format='pdf')
        plt.close()

    # plt.show()


def main(edges_path, nodes_path, write_path):
    G = read_graph(nodes_path, edges_path)
    plot(G, savepath=os.path.join(write_path, 'graphical_model.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating graph for graphical model. The same used in the paper.'
    )
    parser.add_argument(
        '--edges-path', action='store', required=True,
        help='Path edges.json file.'
    )

    parser.add_argument(
        '--nodes-path', action='store', required=True,
        help='Path nodes.json file.'
    )

    parser.add_argument(
        '--write-path', action='store', required=True,
        help='Where to write the .pdf file.'
    )

    args = parser.parse_args()
    main(edges_path=args.edges_path, nodes_path=args.nodes_path, write_path=args.write_path)

