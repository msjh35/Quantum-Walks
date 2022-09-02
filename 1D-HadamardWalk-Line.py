# Walk on Line with Hadamard Coin
# Parameters: Number of qubits, number of steps, number of repetitions, initial state (coin and position)

import cirq
import random
import numpy as np
from matplotlib import pyplot as plt
import scipy

number_qubits =10
qubits = cirq.GridQubit.rect(1, number_qubits)
number_steps = 100
sample_number = 100000

coin_qubit = cirq.GridQubit(0, number_qubits)

def initial_state():
# Initialise position vector    
    yield cirq.X.on(cirq.GridQubit(0, 0))
# Initialise coin in symmetric state 
    yield cirq.H.on(coin_qubit)
    yield cirq.S.on(coin_qubit)

def walk_step():
    # Hadamard coin flip 
    yield cirq.H.on(coin_qubit)

    # Implement the Addition Operator/Shift Right
    yield cirq.X.on(coin_qubit)
    for i in range(number_qubits, 0, -1):
        controls = [cirq.GridQubit(0, v) for v in range(number_qubits, i-1, -1)]
        yield cirq.X.on(cirq.GridQubit(0, i-1)).controlled_by(*controls)
        if (i > 1):
            yield cirq.X.on(cirq.GridQubit(0, i-1))
    yield cirq.X.on(coin_qubit)

    # Implement the Substraction Operator/Shift Left
    for i in range(1, number_qubits+1):
        controls = [cirq.GridQubit(0, v) for v in range(number_qubits, i-1, -1)]
        yield cirq.X.on(cirq.GridQubit(0, i-1)).controlled_by(*controls)
        if (i < number_qubits):
            yield cirq.X.on(cirq.GridQubit(0, i))

def generate_walk(number_qubits, number_steps, sample_number):
    circuit = cirq.Circuit()
    circuit.append(initial_state())
    for j in range(0, number_steps):
        circuit.append(walk_step())
    circuit.append(cirq.measure(*qubits, key='x'))
    print(circuit)

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=sample_number)
    final = result.histogram(key='x')
    return final

final = generate_walk(number_qubits, number_steps, sample_number)

def graph(final):
    x_arr = list(final.keys())
    y_arr = [dict(final)[j] for j in dict(final).keys()]

    # Order elemnents of x in ascending order
    zipped_lists = zip(x_arr, y_arr)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_y = [element for _, element in sorted_zipped_lists]
    sorted_x = [element for element,_ in sorted_zipped_lists]

    # Normalise y values (probability)
    summ = sum(sorted_y)
    prob = [i/summ for i in sorted_y]
    
    scaled_x = [x-512 for x in sorted_x]
    plt.plot(scaled_x, prob)
    plt.scatter(scaled_x, prob, s=1)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.show()

graph(final)