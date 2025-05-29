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

def create_initial_state(transition_matrix,coefficients=None,nodes=None,extended_phases=None,link_phases=None):
    """Creates a suitable initial state from the transition matrix.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
        coefficients: List of coefficients for the linear combination of the psi_i states. Default: 1/np.sqrt(len(nodes)).
        nodes: List of nodes corresponding to the psi_i states of the linear combination. Default: all nodes.
        extended_phases: Angles of the extended Szegedy model.
        link_phases: Alias for extended_phases.
    
    Returns:
        initial_state: NumPy tensor of shape (N,) representing the initial state.
        
    Raises:
        Exception: If the transition matrix is not column-stochastic.
        Exception: If coefficients and nodes have different length.
        Exception: If values are provided for both extended_phases and link_phases.
    """
    
    if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
        raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
    
    if extended_phases is not None and link_phases is not None:
        raise ValueError("The declaration of both extended_phases and link_phases is ambiguous. Use only one.")
    
    if link_phases is not None:
        extended_phases = link_phases
    
    N = transition_matrix.shape[0]
    
    if nodes is None:
        nodes = np.arange(N)
    
    if coefficients is None:
        coefficients = [1/np.sqrt(len(nodes)) for _ in range(len(nodes))]
    coefficients = np.array(coefficients)
    
    if len(nodes) != len(coefficients):
        raise Exception('The parameters \'coefficients\' and \'nodes\' must be lists of the same size.')
    
    # Create the list with the coefficients of the linear combination including zeroes.
    coefficients_total = np.zeros([N]).astype(coefficients.dtype)
    coefficients_total[nodes] = coefficients
    coefficients_total = np.reshape(np.array(coefficients_total),(1,N))
    
    # The initial state is created unrolling the matrix that results of the product of the coefficients
    # and the square root of the transition matrix.
    if extended_phases is None:
        initial_state = np.ravel((coefficients_total*np.sqrt(transition_matrix)).T)
    else:
        extended_phases = np.array(extended_phases)
        extended_factors = np.exp(1j*extended_phases).T
        initial_state = np.ravel((coefficients_total*np.sqrt(transition_matrix)*extended_factors).T)
    return initial_state

def create_psi_states(transition_matrix,nodes=None,extended_phases=None,link_phases=None):
    """Creates the psi_i position states from the transition matrix.
    
    Args:
        transition_matrix: Classical column-stochastic transition matrix.
        nodes: Nodes corresponding to the psi_i states:
            -int: Return a single psi_i state.
            -list: Return a batch with the psi_i states.
            -default: Return a batch with all the psi_i states.
        extended_phases: Angles of the extended Szegedy model.
        link_phases: Alias for extended_phases.

    Returns:
        psi_state: NumPy tensor of shape (N,) representing the psi_i state.
        psi_batch: NumPy tensor of shape (N,len(nodes)) representing the batch of psi_i states.
        
    Raises:
        Exception: If the transition matrix is not column-stochastic.
        Exception: If values are provided for both extended_phases and link_phases.
    """
    
    if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
        raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
    
    if extended_phases is not None and link_phases is not None:
        raise ValueError("The declaration of both extended_phases and link_phases is ambiguous. Use only one.")
    
    if link_phases is not None:
        extended_phases = link_phases
    
    N = transition_matrix.shape[0]
    
    if nodes is None:
        nodes = np.arange(N)
    
    if extended_phases is None:
        
        if type(nodes) == int:  # Create a single psi_i state.
            psi_state = np.zeros([N**2])
            psi_state[N*nodes:N*nodes+N] = np.sqrt(transition_matrix[:,nodes])
            return psi_state
        
        else:  # Create a batch with the psi_i states.
            psi_batch = np.zeros([N**2,len(nodes)])
            for index, node in enumerate(nodes):
                psi_batch[N*node:N*node+N,index] = np.sqrt(transition_matrix[:,node])
            return psi_batch
        
    else:
        
        extended_phases = np.array(extended_phases)
        extended_factors = np.exp(1j*extended_phases).T
        
        if type(nodes) == int:  # Create a single psi_i state.
            psi_state = np.zeros([N**2])*1j
            psi_state[N*nodes:N*nodes+N] = np.sqrt(transition_matrix[:,nodes])*extended_factors[:,nodes]
            return psi_state
        
        else:  # Create a batch with the psi_i states.
            psi_batch = np.zeros([N**2,len(nodes)])*1j
            for index, node in enumerate(nodes):
                psi_batch[N*node:N*node+N,index] = np.sqrt(transition_matrix[:,node])*extended_factors[:,node]
            return psi_batch