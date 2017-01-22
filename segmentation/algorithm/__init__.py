from .slic import slic, slic_generator, slic_json_generator
from .slico import slico, slico_generator, slico_json_generator
from .quickshift import quickshift, quickshift_generator,\
                        quickshift_json_generator
from .felzenszwalb import felzenszwalb, felzenszwalb_generator,\
                          felzenszwalb_json_generator


algorithms = {'slic': slic,
              'slico': slico,
              'quickshift': quickshift,
              'felzenszwalb': felzenszwalb}

generators = {'slic': slic_generator,
              'slico': slico_generator,
              'quickshift': quickshift_generator,
              'felzenszwalb': felzenszwalb_generator}

json_generators = {'slic': slic_json_generator,
                   'slico': slico_json_generator,
                   'quickshift': quickshift_json_generator,
                   'felzenszwalb': felzenszwalb_json_generator}
