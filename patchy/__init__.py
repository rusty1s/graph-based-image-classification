from .patchy import PatchySan

from .helper.labeling import identity, betweenness_centrality
from .helper.neighborhood_assembly import neighborhoods_by_weight


labelings = {'identity': identity,
             'betweenness_centrality': betweenness_centrality}

neighborhood_assemblies = {'by_weight': neighborhoods_by_weight}
