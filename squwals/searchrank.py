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

High-level applications of the quantum and semiclassical SearchRank.
"""

import numpy as np
import networkx as nx
from squwals.operators import Reflection, Swap, Oracle, Unitary
from squwals.simulators import core_szegedy_simulator, mixed_state_simulator
from squwals.applications import semiclassical_szegedy_simulator

def build_google_matrix_nx(graph,alpha=0.85):
    """Function that creates the Google matrix from a networkX graph.
    
    Args:
        graph: networkX directed graph.
        alpha: damping parameter.
    
    Returns:
        google_matrix: Google matrix.
    """
    
    H = nx.to_numpy_array(graph)
    
    N = H.shape[0]
    
    for a in range(N):
        for b in range(N):
            if H[a,b] != 0:
                H[a,b] = 1
    
    # Create an auxiliary graph having removed multiple edges.
    graph_aux = nx.from_numpy_array(H,create_using=nx.DiGraph)
    
    # For our software the Google matrix is the transpose of the one provided by networkX.
    google_matrix = nx.google_matrix(graph_aux,alpha).T
    
    return np.squeeze(np.asarray(google_matrix))

def semiclassical_pagerank(semiclassical_matrices,time_steps=1000):
    """Semiclassical PageRank simulator. It calculates the limiting distribution of a set
       of semiclassical Google matrices.

    Args:
        semiclassical_matrices: A tensor with the semiclassical matrices at each time step.
        time_steps: Number of steps of the classical walk.
    
    Returns:
        distributions: Limiting distribution for each of the semiclassical matrices.
    """
    
    N = semiclassical_matrices.shape[1]
    
    distributions = np.ones([semiclassical_matrices.shape[0],N,1])/N
    
    for t in range(time_steps):
        distributions = semiclassical_matrices @ distributions
    
    return np.squeeze(distributions)

class QuantumSearchrankResults():
    """Class for storing the results of the quantum searchrank.

    Attributes:
        instantaneous_quantum_searchrank: Instantaneous quantum searchranks.
        quantum_probabilities: Probabilities of measuring the marked nodes at each time step.
        quantum_searchrank: Quantum searchrank distribution at the reference time.
    """
    
    def __init__(self):
        """Initializes the class for the results."""
        self.instantaneous_quantum_searchrank = None
        self.quantum_probabilities = None
        self.quantum_searchrank = None

class SemiclassicalSearchrankResults():
    """Class for storing the results of the semiclassical searchrank.

    Attributes:
        semiclassical_matrices: Semiclassical transition matrices at each time step.
        instantaneous_semiclassical_searchrank: Instantaneous semiclassical searchranks.
        semiclassical_probabilities: Probabilities of measuring the marked nodes for the semiclassical searchrank.
        semiclassical_searchrank: Semiclassical searchrank distribution at the reference time.
        instantaneous_randomized_searchrank: Instantaneous randomized searchranks.
        randomized_probabilities: Probabilities of measuring the marked nodes for the randomized searchrank.
        randomized_searchrank: Randomized searchrank distribution at the reference time.
    """
    
    def __init__(self):
        """Initializes the class for the results."""
        self.semiclassical_matrices = None
        self.instantaneous_semiclassical_searchrank = None
        self.semiclassical_probabilities = None
        self.semiclassical_searchrank = None
        self.instantaneous_randomized_searchrank = None
        self.randomized_probabilities = None
        self.randomized_searchrank = None

def quantum_searchrank(google_matrix,time_steps=50,marked_nodes=[],unitary_operator=None,measure='Y'):
    """Quantum SearchRank simulator.

    Args:
        google_matrix: Column-stochastic transition matrix.
        time_steps: Number of steps of the quantum walk.
        marked_nodes: A list with the nodes to mark.
        unitary_operator: Unitary operator model (optional). Default: U=SQ1RSQ1R.
        measure: An intenger indicating the register for measuring. Default: second register.
    
    Returns:
        results: Object with the results.
    """
    
    results = QuantumSearchrankResults()
    N = np.shape(google_matrix)[0]  # Size of the graph.
    M = len(marked_nodes)  # Number of marked nodes.
    
    ref_time = int(np.round(np.sqrt(N/M)))  # Reference time of measurement.
    
    # If no unitary operator is provided, it creates the default SearchRank operator.
    if unitary_operator is None:
        R = Reflection(google_matrix)
        S = Swap()
        Q1 = Oracle(1,marked_nodes)
        unitary_operator = Unitary([R,Q1,S,R,Q1,S])
    
    state = np.expand_dims(np.sqrt(google_matrix)/np.sqrt(N), axis=2)  # Tensorize the state.
    
    # Simulate the quantum walk and calculate the quantum SearchRank.
    results.instantaneous_quantum_searchrank = np.squeeze(core_szegedy_simulator(unitary=unitary_operator,state=state,time_steps=time_steps,measure=measure))
    
    results.quantum_probabilities = np.sum(results.instantaneous_quantum_searchrank[:,marked_nodes],axis=1)
    
    results.quantum_searchrank = results.instantaneous_quantum_searchrank[ref_time]
    
    return results

def semiclassical_searchrank(google_matrix,quantum_time_steps=50,marked_nodes=[],batch_size=1,classical_time_steps=2000,unitary_operator=None,measure='Y',monitor=True):
    """Semiclassical SearchRank simulator.

    Args:
        google_matrix: Column-stochastic transition matrix.
        quantum_time_steps: Number of quantum time steps of the semiclassical walk.
        marked_nodes: A list with the nodes to mark.
        batch_size: Number of states being vectorized at a batch.
        classical_time_steps: Number of classical time steps of the semiclassical walk.
        unitary_operator: Unitary operator model (optional). Default: U=SQ1RSQ1R.
        measure: An intenger indicating the register for measuring. Default: second register.
        monitor: If True, the number of current simulated states is printed.
    
    Returns:
        results: Object with the results.
    """
    
    results = SemiclassicalSearchrankResults()
    N = np.shape(google_matrix)[0]  # Size of the graph.
    M = len(marked_nodes)  # Number of marked nodes.
    
    ref_time = int(np.round(np.sqrt(N/M)))  # Reference time of measurement.
    
    # If no unitary operator is provided, it creates the default SearchRank operator.
    if unitary_operator is None:
        R = Reflection(google_matrix)
        S = Swap()
        Q1 = Oracle(1,marked_nodes)
        unitary_operator = Unitary([R,Q1,S,R,Q1,S])
    
    # Simulate the semiclassical walk and calculate the semiclassical SearchRank.
    results.semiclassical_matrices = semiclassical_szegedy_simulator(transition_matrix=google_matrix,time_steps=quantum_time_steps,unitary=unitary_operator,batch_size=batch_size,measure=measure,monitor=monitor)
    
    results.instantaneous_semiclassical_searchrank = semiclassical_pagerank(results.semiclassical_matrices,classical_time_steps)
    
    results.semiclassical_probabilities = np.sum(results.instantaneous_semiclassical_searchrank[:,marked_nodes],axis=1)
    
    results.semiclassical_searchrank = results.instantaneous_semiclassical_searchrank[ref_time]
    
    results.instantaneous_randomized_searchrank = mixed_state_simulator(basis_simulations=results.semiclassical_matrices,coefficients=np.ones(N)/N)
    
    results.randomized_probabilities = np.sum(results.instantaneous_randomized_searchrank[:,marked_nodes],axis=1)
    
    results.randomized_searchrank = results.instantaneous_randomized_searchrank[ref_time]
    
    return results