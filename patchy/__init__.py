from .patchy import PatchySan

from .labeling import identity, weight_to_first_label, betweenness_centrality


labelings = {'identity': identity,
             'betweenness_centrality': betweenness_centrality,
             'weight_to_first_label': weight_to_first_label}
