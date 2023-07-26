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

Source code of the package.
"""

import numpy as np
import gc

## Operator classes ---------------------------------------------------------##

class Unitary():
    """Unitary operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
        class_type: Kind of class.
    """
    
    class_type = 'unitary'
    
    def __init__(self,operators=None,name=None):
        """Initializes the unitary model.

        Args:
            operators: A list of operators to create directly the unitary.
            name: Custom name for the unitary operator.
        """
        
        self.string = ''
        self.info_string = 'Custom unitary:'
        if name is not None:
            self.info_string += f' {name}'
        if operators is None:  # Initialize an empty unitary.
            self.operators = []
            self.info_index = 0
        else:  # Use the list to initialize the unitary.
            self.operators = operators
            for op_index, operator in enumerate(operators):
                self.string = operator.string + self.string
                self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string
            self.info_index = op_index
    
    def __str__(self):
        """Function for printing the unitary string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operators in the unitary model."""
        
        print(self.info_string)
    
    def append(self,operator):
        """Append a new operator class to the list of operators."""
        
        self.operators.append(operator)
        self.string = operator.string + self.string
        self.info_string += f'\n {self.info_index+1} - ' + operator.info_string
        self.info_index += 1
    
    def operate(self,state,mode='vector',protect=True):
        """Apply the operators over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
              protect: Whether to protect or no the initial state variable
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            if protect: state = state.copy()  # Protect the initial state variable in this mode
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        for operator in self.operators:  # Operate over the state.
            state = operator.operate(state,mode='tensor')
            
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        if unitary_2.class_type == 'unitary':
            return Unitary(unitary_2.operators + self.operators)
        else:
            return Unitary([unitary_2] + self.operators)
        

class Swap():
    """Swap operator.

    Attributes:
      string: A string representing the operator in an algebraic form.
      info_string: A string with the information of the operator.
      class_type: Kind of class.
    """
    
    string = 'S'
    info_string = 'Swap'
    class_type = 'operator'
    
    def __str__(self):
        """Function for printing the operator string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operator."""
        
        print(self.info_string)
    
    def operate(self,state,mode='vector'):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        state = np.transpose(state,axes=(1,0,2))  # The swap operation is done transposing the matrix state.
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        return Unitary([self]) * unitary_2

class Oracle():
    """Oracle operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        register: An intenger indicating the register for marking.
        marked_nodes: A list with the nodes to mark.
        phase: Angle for the arbitrary phase rotation.
        factor: Number to multiply the marked elements.
        class_type: Kind of class.
    """
    
    string = 'Q'
    info_string = 'Oracle'
    class_type = 'operator'
    
    def __init__(self,register,marked_nodes,phase=None,name=None):
        """Initializes the oracle operator.

        Args:
            register: An intenger indicating the register for marking: 1 or 2.
            marked_nodes: A list with the nodes to mark.
            phase: Angle for the arbitrary phase rotation (optional).
            name: Custom name for the oracle operator.
        """
        
        if name is not None:
            self.info_string += f' {name}'
        self.register = register
        if register == 1: self.string = self.string + '\N{SUBSCRIPT ONE}'
        if register == 2: self.string = self.string + '\N{SUBSCRIPT TWO}'
        self.info_string += f': Register {register}'
        self.marked_nodes = marked_nodes
        self.info_string += f': nodes {marked_nodes}'
        self.phase = phase
        if phase is None:  # If no phase is provided it simply inverts the sign.
            self.factor = -1
        else:
            self.factor = np.exp(1j*phase)
            self.info_string += f', phase = {phase:.2f}'
    
    def __str__(self):
        """Function for printing the operator string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operator."""
        
        print(self.info_string)
    
    def operate(self,state,mode='vector',protect=True):
        """Apply the operator over an initial state.
        
        Warning: an oracle can modify the original input.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
            protect: Whether to protect or no the initial state variable
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            if protect: state = state.copy()  # Protect the initial state variable in this mode
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        if self.phase is not None: # If there is a complex phase, the initial state must be complex
            if state.dtype != 'complex128':
                state = state.astype('complex128')
        # Mark the elements multiplying by the factor.
        if self.register == 1:
            state[:,self.marked_nodes] = state[:,self.marked_nodes]*self.factor
        else:
            state[self.marked_nodes,:] = state[self.marked_nodes,:]*self.factor
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        return Unitary([self]) * unitary_2

class Reflection():
    """Reflection operator.

    Attributes:
        string: A string representing the operator in an algebraic form.
        info_string: A string with the information of the operator.
        psi_matrix: Matrix Psi needed for the reflection.
        apr_factor: Factor used in the arbitrary phase rotation.
        apr_phase: Phase for the arbitrary phase rotation.
        extended_factors: Factors to multiply the psi_matrix in the extended Szegedy model.
        extended_phases: Angles of the extended Szegedy model.
        class_type: Kind of class.
    """
    
    string = 'R'
    info_string = 'Reflection'
    class_type = 'operator'
    
    def __init__(self,transition_matrix,apr_phase=None,extended_phases=None,name=None):
        """Initializes the reflection operator.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            apr_phase: Phase for the arbitrary phase rotation (optional).
            extended_phases: Matrix with the phases of the extended Szegedy model (optional).
            name: Custom name for the reflection operator.
        """
        
        if name is not None:
            self.info_string += f' {name}'
        N = np.shape(transition_matrix)[1];  # Size of the graph.
        self.psi_matrix = np.sqrt(transition_matrix).reshape(N,N,1)  # Creates the psi_matrix from the transition matrix.
        if extended_phases is not None:
            self.extended_phases = np.array(extended_phases)
            self.extended_factors = np.expand_dims(np.exp(1j*self.extended_phases).T,axis=2)
            self.psi_matrix = self.psi_matrix*self.extended_factors  # The psi_matrix is modified by the extended phases.
            self.info_string += ' (extended model)'
        if apr_phase is None:
            self.apr_factor = 2  # The default apr factor is 2.
        else:
            self.apr_phase = apr_phase
            self.apr_factor = 1-np.exp(1j*apr_phase)
            self.info_string += f': apr_phase = {apr_phase:.2f}'
            
    def __str__(self):
        """Function for printing the operator string."""
        
        return self.string
    
    def info(self):
        """Print the information of the operator."""
        
        print(self.info_string)
    
    def operate(self,state,mode='vector'):
        """Apply the operator over an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            state: Final vector after the operations.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        # Apply the operations corresponding to the reflection.
        C_matrix = np.sum(self.psi_matrix * state, axis=0, keepdims=True);
        state_parallel = self.psi_matrix*C_matrix
        state = self.apr_factor*state_parallel - state
        
        if mode == 'vector':  # Detensorize to the original shape.
            if dimension == 1:
                state = np.transpose(state,axes=(1,0,2)).reshape(N**2)
            else:
                state = np.transpose(state,axes=(1,0,2)).reshape([N**2,shape[1]])
        
        return state
    
    def __mul__(self,unitary_2):
        """Multiplication of two unitary operators in an algebraic form."""
        
        return Unitary([self]) * unitary_2

class Measurement():
    """Class to obtain the probability distributions.

    Attributes:
        info_string: A string with the information of the operator.
        register: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
    """
    
    info_string = 'Measurement'
    
    def __init__(self,register):
        """Initializes the oracle operator.

        Args:
            register: Register to measure:
                -'X' or 'x' or 1: register 1.
                -'Y' or 'y' or 2: register 2.
                -'XY' or 'xy' or 12: both registers.
        """
        
        self.register = register
        self.info_string += f': Register {register}'
    
    def info(self):
        """Print the information of the measurement."""
        
        print(self.info_string)
    
    def operate(self,state,mode='vector'):
        """Measure an initial state.
        
        Args:
            state: Initial vector state.
            mode: A string to indicate whether the initial state is a vector or
              a matrix state.
        
        Returns:
            measure: Probability distribution after the measurement.
              -If both registers are being measured, a tuple with 2 elements is returned.
        """
        
        if mode == 'vector':  # Tensorize into a matrix state.
            shape = state.shape
            dimension = len(shape)
            N = int(np.sqrt(shape[0]))
            if dimension == 1:
                state = np.expand_dims(state.reshape(N,N).T, axis=2)
            else:
                state = np.transpose(state.reshape([N,N,state.shape[1]]),axes=(1,0,2))
                
        # Perform the measurement.
        abs_sq_state = np.abs(state)**2
        if self.register == 'Y' or self.register == 'y' or self.register == 2:
            measure = np.sum(abs_sq_state,axis=1)
            if dimension == 1:
                measure = np.squeeze(measure)
        elif self.register == 'X' or self.register == 'x' or self.register == 1:
            measure = np.sum(abs_sq_state,axis=0)
            if dimension == 1:
                measure = np.squeeze(measure)
        else:
            measure_y = np.sum(abs_sq_state,axis=1)
            measure_x = np.sum(abs_sq_state,axis=0)
            if dimension == 1:
                measure_x = np.squeeze(measure_x)
                measure_y = np.squeeze(measure_y)
            measure = (measure_x, measure_y)
        
        return measure



## Default unitaries --------------------------------------------------------##

class SingleUnitary(Unitary):
    """Single unitary Szegedy operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
    """
    
    def __init__(self,transition_matrix,apr_phase=None,extended_phases=None,name=None):
        """Initializes the unitary model.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            apr_phase: Phase for the arbitrary phase rotation (optional).
            extended_phases: Matrix with the phases of the extended Szegedy model (optional).
            name: Custom name for the unitary operator.
        """
        
        self.string = 'SR'
        self.operators = []
        self.operators.append(Reflection(transition_matrix,apr_phase=apr_phase,extended_phases=extended_phases))
        self.operators.append(Swap())
        self.info_string = 'Single Szegedy unitary:'
        if name is not None:
            self.info_string += f' {name}'
        for op_index, operator in enumerate(self.operators):
            self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string

class DoubleUnitary(Unitary):
    """Double unitary Szegedy operator model.

    Attributes:
        string: A string representing the unitary operator in an algebraic form.
        info_string: A string with the information of all the operators inside the unitary.
    """
    
    def __init__(self,transition_matrix,apr_phase_1=None,apr_phase_2=None,extended_phases_1=None,extended_phases_2=None,name=None):
        """Initializes the unitary model.

        Args:
            transition_matrix: Classical column-stochastic transition matrix.
            apr_phase_1: Phase for the arbitrary phase rotation of the first reflection (optional).
            apr_phase_2: Phase for the arbitrary phase rotation of the second reflection (optional).
            extended_phases_1: Matrix with the phases of the extended Szegedy model of the first reflection (optional).
            extended_phases_2: Matrix with the phases of the extended Szegedy model of the second reflection (optional).
            name: Custom name for the unitary operator.
        """
        
        self.string = 'SRSR'
        self.operators = []
        if apr_phase_1 is None and apr_phase_2 is None and extended_phases_1 is None and extended_phases_2 is None:
            R1 = Reflection(transition_matrix)
            R2 = R1
        else:
            R1 = Reflection(transition_matrix,apr_phase=apr_phase_1,extended_phases=extended_phases_1)
            R2 = Reflection(transition_matrix,apr_phase=apr_phase_2,extended_phases=extended_phases_2)
        S = Swap()
        self.operators.append(R1)
        self.operators.append(S)
        self.operators.append(R2)
        self.operators.append(S)
        self.info_string = 'Double Szegedy unitary:'
        if name is not None:
            self.info_string += f' {name}'
        for op_index, operator in enumerate(self.operators):
            self.info_string = self.info_string + f'\n {op_index+1} - ' + operator.info_string



        

## Initial state ------------------------------------------------------------##

def create_initial_state(transition_matrix):
    """Creates a suitable initial state from the transition matrix.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
    
    Returns:
        initial_state: NumPy tensor of shape (N,) representing the initial state.
    """
    
    N = transition_matrix.shape[0]
    # The initial state is created unrolling the matrix transition_matrix.
    initial_state = np.sqrt(np.ravel(transition_matrix.T))/np.sqrt(N)
    return initial_state




## Simulators ---------------------------------------------------------------##

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




## High-level applications --------------------------------------------------##

def semiclassical_szegedy_simulator(transition_matrix,time_steps=100,unitary='single',batch_size=1,measure=1,monitor=True,load_file=None,checkpoints=False):
    """Simulator of the semiclassical matrices of the semiclassical Szegedy walk.

    Args:
        transition_matrix: Classical column-stochastic transition matrix.
        time_steps: Number of quantum time steps.
        unitary: unitary operator model (optional). Default is 'single', so U = SR, constructing R from transition_matrix.
        batch_size: Number of states being vectorized at a batch.
        measure: Register to measure:
            -'X' or 'x' or 1: register 1.
            -'Y' or 'y' or 2: register 2.
            -'XY' or 'xy' or 12: both registers.
        monitor: If True, the number of current simulated states is printed.
        # load_file: file path to load intermediate results (TODO).
        # checkpoints: if True, intermediate results are saved after each batch simulation (TODO).
    
    Returns:
        semiclassical_matrices: A tensor with the probability distributions at each time step.
            -If both registers are being measured, a tuple with 2 elements is returned.
    """
    
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
        unitary_operator: unitary operator model (optional). Default: U=SRSR.
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