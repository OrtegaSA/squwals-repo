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

High-level applications of the quantum walk.
"""

import numpy as np
import gc
from squwals.operators import SingleUnitary, DoubleUnitary
from squwals.simulators import core_szegedy_simulator

def semiclassical_szegedy_simulator(transition_matrix,time_steps=100,unitary='single',batch_size=1,measure=1,monitor=True):
    """Simulator of the semiclassical matrices of the semiclassical Szegedy walk.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
        time_steps: Number of quantum time steps.
        unitary: Unitary operator model (optional). Default is 'single', so U = SR, constructing R from transition_matrix.
        batch_size: Number of states being vectorized at a batch.
        measure: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
        monitor: If True, the number of current simulated states is printed.
    
    Returns:
        semiclassical_matrices: A tensor with the semiclassical matrices at each time step.
            -If both registers are being measured, a tuple with 2 elements is returned.
    
    Raises:
        Exception: If the transition matrix is not column-stochastic.
    """
    
    if np.allclose(np.sum(transition_matrix,axis=0),np.ones(np.shape(transition_matrix)[1])) != True:
        raise Exception('The transition matrix is not column-stochastic. See tutorial: https://github.com/OrtegaSA/squwals-repo/tree/main/Tutorials')
    
    # Create the unitary operator if one of the two default options is chosen.
    if unitary == 'single':
        unitary = SingleUnitary(transition_matrix)
    elif unitary == 'double':
        unitary = DoubleUnitary(transition_matrix)
    
    N = transition_matrix.shape[0]  # Size of the graph.
    
    # Create the tensors for saving the results.
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        semiclassical_matrices_x = np.zeros([time_steps+1,N,0])
        semiclassical_matrices_y = np.zeros([time_steps+1,N,0])
        current_dim = semiclassical_matrices_x.shape[2]
    else:
        semiclassical_matrices = np.zeros([time_steps+1,N,0])
        current_dim = semiclassical_matrices.shape[2]
    
    nodes = np.arange(N)
    # Create the batchs and simulate the quantum walk.
    for epoch, batch_node in enumerate(range(0,N,batch_size)):
        
        # Create the batch of the psi_i states.
        transition_matrix_batch = transition_matrix[:,batch_node:batch_node+batch_size]
        nodes_batch = nodes[batch_node:batch_node+batch_size]
        batch_dim = transition_matrix_batch.shape[1]
        psi_batch = np.zeros([N,N,batch_dim])
        for index, node in enumerate(nodes_batch):
            psi_batch[:,node,index]  = np.sqrt(transition_matrix[:,node]);
        
        if monitor: print('Number of simulated nodes = ',current_dim)
        if monitor: print('Epoch = ',epoch+1,', Current nodes =',batch_node+1,'-',batch_node+psi_batch.shape[2])
        
        # Simulate the batch.
        batch_results = core_szegedy_simulator(unitary=unitary,state=psi_batch,time_steps=time_steps,measure=measure)
        
        # Concatenate the batch results to the current results.
        if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
            semiclassical_matrices_x = np.concatenate((semiclassical_matrices_x,batch_results[0]),axis=2)
            semiclassical_matrices_y = np.concatenate((semiclassical_matrices_y,batch_results[1]),axis=2)
            current_dim = semiclassical_matrices_x.shape[2]
        else:
            semiclassical_matrices = np.concatenate((semiclassical_matrices,batch_results),axis=2)
            current_dim = semiclassical_matrices.shape[2]
        
        epoch += 1
        
        if monitor: print('-------------------------')
        
        # Eliminate the batch from the memory.
        del psi_batch
        gc.collect()
        
    if monitor: print('Number of simulated nodes = ',current_dim)
    
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        return semiclassical_matrices_x, semiclassical_matrices_y
    else:
        return semiclassical_matrices

class Qpr():
    """Class for storing the results of the quantum pagerank.

    Attributes:
        instantaneous: Instantaneous quantum pageranks.
        averaged: Averaged quantum pageranks.
        std: Standard deviation of the averaged quantum pageranks.
    """
    
    def __init__(self):
        """Initializes the class for the results."""
        
        self.instantaneous = None
        self.averaged = None
        self.std = None

def quantum_pagerank(google_matrix,time_steps=1000,apr_phase_1=None,apr_phase_2=None,unitary=None,measure=2):
    """Quantum PageRank simulator.

    Args:
        google_matrix: Column-stochastic transition matrix.
        time_steps: Number of steps of the quantum walk.
        apr_phase_1: Arbitrary phase rotation of the first reflection (optional).
        apr_phase_2: Arbitrary phase rotation of the second reflection (optional).
        unitary: Unitary operator model (optional). Default: U=SRSR.
        measure: An intenger indicating the register for measuring. Default: second register.
    
    Returns:
        results: Object with the results.
    """
    
    results = Qpr()
    N = np.shape(google_matrix)[0]  # Size of the graph.
    
    # If no unitary operator is provided, it creates the default PageRank operator.
    if unitary is None:
        unitary = DoubleUnitary(google_matrix,apr_phase_1,apr_phase_2)
    
    state = np.expand_dims(np.sqrt(google_matrix)/np.sqrt(N), axis=2)  # Tensorize the state.
    
    # Simulate the quantum walk and calculate the quantum PageRank.
    results.instantaneous = np.squeeze(core_szegedy_simulator(unitary=unitary,state=state,time_steps=time_steps,measure=measure))
    results.averaged = np.mean(results.instantaneous,axis=0)
    results.std = np.std(results.instantaneous,axis=0)
    
    return results
