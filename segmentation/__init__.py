from .feature_extraction import feature_extraction, NUM_FEATURES
from .adjacency import adjacency_unweighted,\
                       adjacency_euclidean_distance


adjacencies = {'unweighted': adjacency_unweighted,
               'euclidean_distance': adjacency_euclidean_distance}
