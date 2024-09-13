import matplotlib.pyplot as plt
import networkx as nx



G = nx.path_graph(10)
shells = [[0], [1, 2, 3], [4,5], [6,7,8,9]]
#pos = nx.shell_layout(G, shells)
#pos = nx.spring_layout(G)
#pos = nx.kamada_kawai_layout(G)
pos = nx.circular_layout(G)

plt.figure(figsize=(8, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')


plt.show()
