from .slic import (slic, slic_generator)
from .slic import (slico, slico_generator)

algorithms = {
    'slic': slic,
    'slico': slico,
}

generators = {
    'slic': slic_generator,
    'slico': slico_generator,
}
