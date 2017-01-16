from .slic import (slic, slic_generator)
from .slic import (slico, slico_generator)

slic_algorithms = {
    'slic': slic,
    'slico': slico,
}

slic_generators = {
    'slic': slic_generator,
    'slico': slico_generator,
}
