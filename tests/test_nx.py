import matplotlib.pyplot as plt
import networkx as nx

G = nx.random_lobster(10, 0.9, 0.9, seed=0)

pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, with_labels=True, node_size=200, node_color="#1260A0", edge_color="gray",
        node_shape="o", font_size=7, font_color="white")


plt.show()