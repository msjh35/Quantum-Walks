# 2D Hadamard Walk using the Simulator.simulate() function from Cirq

import cirq
import random
import numpy as np
from matplotlib import pyplot as plt
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.colors as cls
import matplotlib.cm as cm    
from matplotlib import cm
import matplotlib.colors   
import matplotlib.ticker as ticker
from matplotlib import rcParams

plot3D = True

number_qubits_x = 8
number_qubits_y = 8
qubits_x = cirq.GridQubit.rect(1, number_qubits_x, top=0)
qubits_y = cirq.GridQubit.rect(1, number_qubits_y, top=1)

number_states_total = 2**(number_qubits_x+number_qubits_y+1+1)
k_total = number_states_total/4

number_states = 2**(number_qubits_x)

number_steps = 50

coin_qubit_x = cirq.GridQubit(0, number_qubits_x)
coin_qubit_y = cirq.GridQubit(1, number_qubits_y)

def initial_state():
#Initialise position vector    
    yield cirq.X.on(cirq.GridQubit(0, 0))
    yield cirq.X.on(cirq.GridQubit(1, 0))
# Initialise in symmetric state 
    yield cirq.X.on(coin_qubit_y)
    yield cirq.H.on(coin_qubit_x)
    yield cirq.S.on(coin_qubit_x)
    yield cirq.H.on(coin_qubit_y)
    yield cirq.S.on(coin_qubit_y)

def walk_step_x():
    # Hadamard coin flip 
    yield cirq.H(coin_qubit_x)
    #Right
    yield cirq.X(coin_qubit_x)
    for i in range(number_qubits_x, 0, -1):
        controls_x = [cirq.GridQubit(0, v) for v in range(number_qubits_x, i-1, -1)]
        yield cirq.X(cirq.GridQubit(0, i-1)).controlled_by(*controls_x)
        if (i > 1):
            yield cirq.X(cirq.GridQubit(0, i-1))
    yield cirq.X.on(coin_qubit_x)
    #Left
    for i in range(1, number_qubits_x+1):
        controls_x = [cirq.GridQubit(0, v) for v in range(number_qubits_x, i-1, -1)]
        yield cirq.X(cirq.GridQubit(0, i-1)).controlled_by(*controls_x)
        if (i < number_qubits_x):
            yield cirq.X(cirq.GridQubit(0, i))

def walk_step_y():
    # Hadamard coin flip 
    yield cirq.H(coin_qubit_y)
    #Right
    yield cirq.X(coin_qubit_y)
    for i in range(number_qubits_y, 0, -1):
        controls_y = [cirq.GridQubit(1, v) for v in range(number_qubits_y, i-1, -1)]
        yield cirq.X(cirq.GridQubit(1, i-1)).controlled_by(*controls_y)
        if (i > 1):
            yield cirq.X(cirq.GridQubit(1, i-1))
    yield cirq.X(coin_qubit_y)
    #Left
    for i in range(1, number_qubits_y+1):
        controls_y = [cirq.GridQubit(1, v) for v in range(number_qubits_y, i-1, -1)]
        yield cirq.X(cirq.GridQubit(1, i-1)).controlled_by(*controls_y)
        if (i < number_qubits_y):
            yield cirq.X(cirq.GridQubit(1, i))

def generate_walk(number_qubits_x, number_qubits_y, number_steps):
    circuit = cirq.Circuit()
    circuit.append(initial_state())
    for j in range(0, number_steps):
        circuit.append(walk_step_x())
        circuit.append(walk_step_y())
    state = circuit.final_state_vector(qubit_order=[cirq.GridQubit(0,8), cirq.GridQubit(1,8), cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3), cirq.GridQubit(0,4), cirq.GridQubit(0,5), cirq.GridQubit(0,6), cirq.GridQubit(0,7), cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3), cirq.GridQubit(1,4), cirq.GridQubit(1,5), cirq.GridQubit(1,6), cirq.GridQubit(1,7)])
    prob=np.square((np.abs(state)))

    for i in range(0, int(k_total)):  
        prob[i]=prob[i] + prob[i+int(k_total)] + prob[i+2*int(k_total)] + prob[i+3*int(k_total)]
    index = list(range(int(k_total),number_states_total))
    prob = np.delete(prob,index)

    y = []
    for i in range(0,number_states):
        for j in range(0, number_states):
            y.append(j)

    x = []
    for i in range(0,number_states):
        for j in range(0, number_states):
            x.append(i)
    return prob, x, y

prob, x, y = generate_walk(number_qubits_x, number_qubits_y, number_steps)
prob = prob.tolist()
non_zero = np.nonzero(prob)[0] 

prob = [prob[i] for i in non_zero]
x = [x[i] for i in non_zero]
y = [y[i] for i in non_zero]

x = [i-128 for i in x]
y = [i-128 for i in y]

res_greater = [index for index, val in enumerate(x) if val > 40]
x = np.delete(x,res_greater)
y = np.delete(y,res_greater)
prob = np.delete(prob,res_greater)

res_lower = [index for index, val in enumerate(x) if val < -40]
x = np.delete(x,res_lower)
y = np.delete(y,res_lower)
prob = np.delete(prob, res_lower)

greater_y = [index for index, val in enumerate(y) if val > 40]
y = np.delete(y, greater_y)
x = np.delete(x, greater_y)
prob = np.delete(prob, greater_y)

lower_y = [index for index, val in enumerate(y) if val < -40]
y = np.delete(y,lower_y)
x = np.delete(x, lower_y)
prob = np.delete(prob, lower_y)

def graph(prob, x, y):
    z = np.zeros(len(x))   
    dx = np.ones(len(x))  
    dy = np.ones(len(y))  
    dz = prob

    fig_3d_bar = plt.figure()  
    ax = fig_3d_bar.add_subplot(111, projection='3d')
    norm = cls.Normalize() 
    norm.autoscale(prob)
    cmap = cm.ScalarMappable(norm, 'binary') # Choose any colormap
    ax.bar3d(x, y, z, dx, dy, dz, color=cmap.to_rgba(prob))  
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability', rotation=0, labelpad=10)

    #ax.set_zlabel('Probability')
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    plt.show()

graph(prob, x, y)

    
