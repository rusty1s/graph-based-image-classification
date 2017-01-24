from .patchy import PatchySan

from .helper.labeling import identity, betweenness_centrality


labelings = {'identity': identity,
             'betweenness_centrality': betweenness_centrality}
