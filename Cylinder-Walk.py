# Walk on an M x N Cylinder

import cirq
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.colors as cls
import matplotlib.cm as cm  
import braket1     

# Parameters
number_qubits_x = 6
number_qubits_y = 6

number_steps = 100
sample_number = 10000

M = 15 #Size of cylinder in x direction
N = 15 #Size of cylinder in y direction 

# Qubit Register
qubits_x = cirq.GridQubit.rect(1, number_qubits_x, top=0)
qubits_y = cirq.GridQubit.rect(1, number_qubits_y, top=1)
coin_qubit_x = cirq.GridQubit(0, number_qubits_x)
coin_qubit_y = cirq.GridQubit(1, number_qubits_y)

# Define Projection Operators
ket = braket1.ket1
bra = braket1.bra1
I = np.identity(2**(number_qubits_x))

# Periodic boundary for x-co
Px = I - ket(M+1)*bra(M+1) - ket(0)*bra(0) + ket(1)*bra(M+1) + ket(M)*bra(0)

# Boundary for y-co
Py = I - ket(0)*bra(0) - ket(N+1)*bra(N+1) + ket(N-1)*bra(N+1) + ket(2)*bra(0)

class Projectionx(cirq.Gate):
    def __init__(self):
        super(Projectionx, self)
    def _num_qubits_(self):
        return 6
    def _unitary_(self):
        return Px
    def _circuit_diagram_info_(self, args):
        return "Px", "", "", "", "", ""

class Projectiony(cirq.Gate):
    def __init__(self):
        super(Projectiony, self)
    def _num_qubits_(self):
        return 6
    def _unitary_(self):
        return Py
    def _circuit_diagram_info_(self, args):
        return "Py", "", "", "", "", ""

# Define Walk
def initial_state():
#Initialise position vector in (1,1)  
    yield cirq.X.on(cirq.GridQubit(0, 2))
    yield cirq.X.on(cirq.GridQubit(1, 2))

# Initialise both coins in symmetric state 
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

def generate_walk(number_qubits_x, number_qubits_y, number_steps, sample_number):
    Px = Projectionx()
    Py = Projectiony()
    circuit = cirq.Circuit()
    circuit.append(initial_state())
    for j in range(0, number_steps):
        circuit.append(walk_step_x())
        circuit.append(walk_step_y())
        circuit.append(Px.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1),cirq.GridQubit(0, 2), cirq.GridQubit(0, 3),cirq.GridQubit(0, 4),cirq.GridQubit(0, 5)))
        circuit.append(Py.on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1),cirq.GridQubit(1, 2), cirq.GridQubit(1, 3),cirq.GridQubit(1, 4),cirq.GridQubit(1, 5)))
    circuit.append(cirq.measure(*qubits_x,*qubits_y, key='p'))
    circuit.append(cirq.measure(*qubits_x, key='x'))
    circuit.append(cirq.measure(*qubits_y, key='y'))
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=sample_number)
    final = result.multi_measurement_histogram(keys=['x','y'])
    return final

final = generate_walk(number_qubits_x, number_qubits_y, number_steps, sample_number)

def graph(final):
    x1_arr = list(final.keys())
    x2_arr = [dict(final)[j] for j in dict(final).keys()]
    x_co = [x[0] for x in x1_arr]
    y_co = [x[1] for x in x1_arr]    
    summ = sum(x2_arr)
    norm_p = [i/summ for i in x2_arr]
    x_pos = np.array(x_co)  
    y_pos = np.array(y_co)  
    z_pos = np.zeros(len(x_co))    
    dx = np.ones(len(x_pos))  
    dy = np.ones(len(y_pos))  
    dz = norm_p

    fig_3d_bar = plt.figure()  
    ax = fig_3d_bar.add_subplot(111, projection='3d')
    
    # To add colours
    norm = cls.Normalize() 
    norm.autoscale(norm_p)
    cmap = cm.ScalarMappable(norm, 'rainbow') # Choose any colormap

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=cmap.to_rgba(norm_p))  
    ax.set_xlabel('Position $x_1$')
    ax.set_ylabel('Position $x_2$')
    ax.set_zlabel('Probability');
    plt.show()

graph(final)
