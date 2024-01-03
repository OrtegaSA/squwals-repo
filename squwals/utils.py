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

Utils of the packcage.
"""

import numpy as np

def create_initial_state(transition_matrix):
    """Creates a suitable initial state from the transition matrix.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
    
    Returns:
        initial_state: NumPy tensor of shape (N,) representing the initial state.
        
    Raises:
            Exception: If the transition matrix is not column-stochastic.
    """
    
    if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
            raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
    
    N = transition_matrix.shape[0]
    # The initial state is created unrolling the matrix transition_matrix.
    initial_state = np.sqrt(np.ravel(transition_matrix.T))/np.sqrt(N)
    return initial_state