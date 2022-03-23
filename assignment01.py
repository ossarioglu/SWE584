import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt
import datetime as dt
import kshell as ks

print("START =", dt.datetime.now())

G_Random = nx.extended_barabasi_albert_graph(5000,1,0.89,0.06)

G = G_Random

print(nx.info(G))
print("After GRAPH Generation =", dt.datetime.now())

k_list = ks.find_k_shell(G)
ks.list_k_shells(k_list)

k_core_values = k_list[-1][1]
print(k_core_values)

print("After K-Shell Operation =", dt.datetime.now())

nx.write_edgelist(G, "C:\\Users\\Osman Selcuk Sariogl\\Desktop\\BOUN\\SWE580\\Project\\Network_Data\\IMDB\\generated.edges")

# nx.draw(G, with_labels= True)
# plt.show()
