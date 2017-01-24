from .feature_extraction import feature_extraction, NUM_FEATURES
from .adjacency import adjacency_unweighted,\
                       adjacency_euclidean_distance


adjacencies = {'unweighted': adjacency_unweighted,
               'euclid_distance': adjacency_euclidean_distance}
