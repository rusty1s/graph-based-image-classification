from .slic import (slic, slic_generator)
from .slic import (slico, slico_generator)
from .quickshift import (quickshift, quickshift_generator)

algorithms = {
    'slic': slic,
    'slico': slico,
    'quickshift': quickshift,
}

generators = {
    'slic': slic_generator,
    'slico': slico_generator,
    'quickshift': quickshift_generator,
}
