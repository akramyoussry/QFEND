##############################################
"""
This module generates a dataset

"""
##############################################
# preample
import numpy as np
from utilities import Pauli_operators, simulate
import sys
################################################
# retrieve noise profile
noise = int(sys.argv[1])
# meta parameters
name        = "G_1q_X_Z_N%d"%(noise)
################################################
# quantum parameters
dim   = 2                                             # dimension of the system
Omega = 12                                            # qubit energy gap
static_operators  = [0.5*Pauli_operators[3]*Omega]    # drift Hamiltonian
dynamic_operators = [0.5*Pauli_operators[1]]          # control Hamiltonian 
noise_operators   = [0.5*Pauli_operators[3]]          # noise Hamiltonian
initial_states    = [
                     np.array([[0.5,0.5],[0.5,0.5]]),
                     np.array([[0.5,-0.5j],[0.5j,0.5]]),
                     np.array([[1,0],[0,0]])
                    ]                                 # intial state of qubit 
measurement_operators = Pauli_operators[1:2]          # measurement operators
##################################################                          
# simulation parameters
T          = 1                                        # Evolution time
M          = 1024                                     # Number of time steps  
num_ex     = 10000                                    # Number of examples
batch_size = 50                                       # batch size for TF 
##################################################
# noise parameters
K               = 2000                                # Number of realzations
noise_profile   = [noise]                             # Noise type
###################################################
# control parameters
pulse_shape     = "Gaussian"                          # Control pulse shape
num_pulses      = 5                                   # Number of pulses per sequence
#################################################### 
# Generate the dataset
sim_parameters = dict( [(k,eval(k)) for k in ["name", "dim", "Omega", "static_operators", "dynamic_operators", "noise_operators", "measurement_operators", "initial_states", "T", "M", "num_ex", "batch_size", "K", "noise_profile", "pulse_shape", "num_pulses"] ])
simulate(sim_parameters)
####################################################
