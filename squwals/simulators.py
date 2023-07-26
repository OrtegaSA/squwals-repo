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

Simulators of the walks.
"""

import numpy as np

def classical_walk_simulator(transition_matrix,time_steps=100,initial_distribution=None,only_last=False):
    """Simulator of the classical walk.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
        time_steps: Number of steps of the classical walk.
        distribution: NumPy tensor of shape (N,). Initial probability distribution of the walker.
        only_last: If True, only the last step distribution is returned.

    Returns:
        distribution: NumPy tensor of shape (N,). Last step walker distribution if only_last == True.
        probability_distributions: NumPy tensor of shape (time_steps+1, N), distributions of the walker at each time step.
    """
    
    N = np.shape(transition_matrix)[1]  # Size of the graph.
    distribution = initial_distribution
    
    if distribution is None:
        distribution = np.ones([N])/N  # The default initial distribution is the uniform one.
    
    probability_distributions = np.zeros([time_steps+1,N]);
    probability_distributions[0] = distribution;
    
    # Simulate the classical walk.
    for t in range(1,time_steps+1):
        distribution = transition_matrix @ distribution;
        probability_distributions[t] = distribution;
    
    if only_last == True:
        return distribution
    else:
        return probability_distributions

def core_szegedy_simulator(unitary='single',state=None,time_steps=100,measure=1):
    """Core function to simulate Szegedy quantum walk.
    
    Args:
        unitary: Unitary operator model.
        state: Initial batch of states in tensorized form.
        time_steps: Number of steps of the quantum walk.
        measure: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
    
    Returns:
        probability_distributions: A tensor with the probability distributions at each time step.
            -If both registers are being measured, a tuple with 2 elements is returned.
    """
    
    N = np.shape(state)[0]  # Size of the graph.
    batch_size = np.shape(state)[2]
    
    # Crete the tensors for saving the results.
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        probability_distributions_x = np.zeros([time_steps+1,N,batch_size])
        probability_distributions_y = np.zeros([time_steps+1,N,batch_size])
    else:
        probability_distributions = np.zeros([time_steps+1,N,batch_size])
        
    # Measure the probability distribution at time 0.
    abs_sq_state = np.abs(state)**2
    if measure == 'Y' or measure == 'y' or measure == 2:
        probability_distributions[0,:,:] = np.sum(abs_sq_state,axis=1)
    elif measure == 'X' or measure == 'x' or measure == 1:
        probability_distributions[0,:,:] = np.sum(abs_sq_state,axis=0)
    else:
        probability_distributions_y[0,:,:] = np.sum(abs_sq_state,axis=1)
        probability_distributions_x[0,:,:] = np.sum(abs_sq_state,axis=0)
        
    # Time loop
    for time in range(1,time_steps+1):
        state = unitary.operate(state,mode='tensor')  # Apply the quantum evolution.
        # Measure the probability distributions.
        abs_sq_state = np.abs(state)**2
        if measure == 'Y' or measure == 'y' or measure == 2:
            probability_distributions[time,:,:] = np.sum(abs_sq_state,axis=1)
        elif measure == 'X' or measure == 'x' or measure == 1:
            probability_distributions[time,:,:] = np.sum(abs_sq_state,axis=0)
        else:
            probability_distributions_y[time,:,:] = np.sum(abs_sq_state,axis=1)
            probability_distributions_x[time,:,:] = np.sum(abs_sq_state,axis=0)
    # End of the time loop
    
    if measure == 'both' or measure == 'XY' or measure == 'xy' or measure == 12:
        return probability_distributions_x, probability_distributions_y
    else:
        return probability_distributions

def quantum_szegedy_simulator(unitary,initial_state,time_steps=100,measure=1,protect=True):
    """Simulator of the Szegedy quantum walk

    Args:
        unitary: Unitary operator model.
        initial_state: Initial state or batch of states.
        time_steps: Number of steps of the quantum walk.
        measure: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
        protect: Whether to protect or no the initial state variable
    
    Returns:
        probability_distributions: A tensor with the probability distributions at each time step.
            -If both registers are being measured, a tuple with 2 elements is returned.
    """
    
    # We must copy the input because the unitary with oracles can change it, if the oracle is the first operator.
    if protect:
        state = initial_state.copy()
    else:
        state = initial_state
    
    # Tensorize into a matrix state.
    shape = state.shape
    dimension = len(shape)
    N = int(np.sqrt(shape[0]))
    if dimension == 1:
        state = np.expand_dims(state.reshape(N,N).T, axis=2)
    elif dimension == 2:
        state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
        
    # Use the core simulator to simulate the quantum walk and retrieve the results in the appropiate form.
    probability_distributions = core_szegedy_simulator(state=state,time_steps=time_steps,unitary=unitary,measure=measure)
    if dimension == 1:
        if type(probability_distributions) == tuple:
            probability_distributions = tuple(np.squeeze(element) for element in probability_distributions)
        else:
            probability_distributions = np.squeeze(probability_distributions)
    
    return probability_distributions

def mixed_state_simulator(basis_simulations,coefficients):
    """Simulator of the Szegedy quantum walk over mixed states.

    Args:
        basis_simulations: Results of the simulation over the batch of the basis states.
        coefficients: Probability distribution of the mixed state over the basis states.

    Returns:
        probability_distributions: A tensor with the probability distributions at each time step.
    """
    
    M = basis_simulations.shape[-1]  # Number of states that comprise the mixed state.
    coefficients = np.reshape(coefficients,[M])
    # Build the probability distributions from the results of the basis states.
    probability_distributions = basis_simulations @ coefficients
    
    return probability_distributions