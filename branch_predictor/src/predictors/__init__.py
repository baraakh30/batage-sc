# Predictors Package
from .base import BasePredictor, PredictionResult
from .perceptron import PerceptronPredictor
from .bnn import BinaryNeuralNetwork
from .hybrid import HybridPredictor
from .optimized_loabp import OptimizedLOABP, OPTIMIZED_CONFIG_32KB, OPTIMIZED_CONFIG_8KB



from .tage_original import (
    OriginalTAGE,
    TAGE_8C_64KB,
    TAGE_32KB,
    TAGE_8KB,
    TAGE_4KB,
)



from .tage_apex import (
    TAGEApex,
    APEX_64KB,
    APEX_32KB,
    APEX_8KB,
)

from .batage_sc import (
    BATAGE_SC,
    BATAGE_SC_64KB,
    BATAGE_SC_32KB,
    BATAGE_SC_8KB,
)

from .tage_scale import (
    TAGE_SCALE,
    TAGE_SCALE_64KB,
    TAGE_SCALE_32KB,
)

from .tage_smart import (
    TAGE_Smart,
    TAGE_SMART_64KB,
    TAGE_SMART_32KB,
)

from .mpp import (
    MPP,
    MPP_64KB,
    MPP_32KB,
)

from .tage_sc_l import (
    TAGE_SC_L,
    TAGE_SC_L_64KB,
    TAGE_SC_L_32KB,
)

from .tage_loop import (
    TAGE_Loop,
    TAGE_LOOP_64KB,
    TAGE_LOOP_32KB,
)


__all__ = [
    'BasePredictor',
    'PredictionResult',
    'PerceptronPredictor', 
    'BinaryNeuralNetwork',
    'HybridPredictor',
    'OptimizedLOABP',
    'OPTIMIZED_CONFIG_32KB',
    'OPTIMIZED_CONFIG_8KB',

    'OriginalTAGE',
    'TAGE_8C_64KB',
    'TAGE_32KB',
    'TAGE_8KB',
    'TAGE_4KB',

    'TAGEApex',
    'APEX_64KB',
    'APEX_32KB',
    'APEX_8KB',

    'BATAGE_SC',
    'BATAGE_SC_64KB',
    'BATAGE_SC_32KB',
    'BATAGE_SC_8KB',
    'TAGE_SCALE',
    'TAGE_SCALE_64KB',
    'TAGE_SCALE_32KB',
]
