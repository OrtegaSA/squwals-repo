# Copyright 2023 Sergio A. Ortega and Miguel A. Martin-Delgado.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQUWALS: A Szegedy QUantum WALks Simulator.

This is a package with utilities to simulate Szegedy's quantum walk in an
efficient manner, saving time and memory resources.
"""

__version__ = '1.0'

# from squwals.source import *

from squwals.source import Unitary, Swap, Oracle, Reflection, Measurement
from squwals.source import SingleUnitary, DoubleUnitary
from squwals.source import create_initial_state
from squwals.source import classical_walk_simulator, quantum_szegedy_simulator, mixed_state_simulator
from squwals.source import semiclassical_szegedy_simulator, quantum_pagerank

__all__ = [
    'Unitary',
    'Swap',
    'Oracle',
    'Reflection',
    'Measurement',
    'SingleUnitary',
    'DoubleUnitary',
    'create_initial_state',
    'classical_walk_simulator',
    'quantum_szegedy_simulator',
    'mixed_state_simulator',
    'semiclassical_szegedy_simulator',
    'quantum_pagerank']