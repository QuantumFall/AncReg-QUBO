#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:53:28 2020

@author: fallenpao
"""

# from qiskit import *
from qiskit import QuantumCircuit, execute, Aer
import numpy as np
import itertools
import time
import copy
from scipy.optimize import minimize


import os


def loadstat():
    stat = os.stat(__file__)
    print("#########################################################################")
    print(" NaSGDv6 file last modified:", time.ctime(stat.st_mtime))
    print(" Contains function to calculate Hessian matrix.")
    print(" Contains classical optimization with SLSQP.")
    print(" N_a-1 parameters work well to fulfill constraints but produces worse results.")
    print("#########################################################################\n")
    return None

def curr_file_dir():
    tttttt = os.path.dirname(os.path.abspath(__file__))
    return tttttt

# Circuit is a function.

def circuit(nq,nd,angvec):
    c = QuantumCircuit(nq,nq)
    
    ###############################
    # Functions to create 1 and 2 qubit gates
    def oneqbgates(j):
        for i in range(nq):
            angle = angvec[j+i]
            c.ry(angle,i)

        return c

    def twoqbgates(m): # m is the depth
        if m%2 == 1 and nq%2 == 1:   
            for g in range(2):
                for k in range(0,nq-2,2):
                    c.cx(g+k,g+1+k)
        if m%2 == 1 and nq%2 == 0:  
            for g in range(2):
                for k in range(0,nq-g-1,2):
                    c.cx(k+g,k+g+1)



        if m%2 == 0 and nq%2 == 1:
            for g in range(1,-1,-1):
                for k in range(0,nq-2,2):
                    c.cx(g+k,g+1+k)
                    
        if m%2 == 0 and nq%2 == 0:
            for g in range(1,-1,-1):
                for k in range(0,nq-g-1,2):
                    c.cx(g+k,g+1+k)
            
        c.barrier()
        return c  

    def mes():
        for i in range(nq):
            c.measure(i,i)
        return c
    
    ###############################
    # Creating the circuit is just calling the above functions
    
    # Put all qubits in Hadamard
    for i in range(nq):
        c.h(i)
    
    # Generates 3 layers of gates
    j=0
    for i in range(nd):
        twoqbgates(i)
        oneqbgates(j)
        c.barrier()
        
        j = j+nq
       
    # Add measurement at the end
    mes()
    return c


def cdict(circ,measurementcounts):
    job = execute(circ,Aer.get_backend('qasm_simulator'),shots=measurementcounts)
    cdict = job.result().get_counts(circ)
    return cdict

def countslist(mesresults):
    nq = len(list(mesresults)[0])
    
    clist = np.zeros(2**nq)
    
    for key in mesresults:
        pos = int(key,2)
        clist[pos] = mesresults[key]
        
    return clist +1

def getproj(mesresults,na,nr):
    
    nstates_anc = 2**na
    
    counts = np.zeros(2**(na+nr))
    
    for key in mesresults:
        pos = int(key,2)
        counts[pos] = mesresults[key] 
    
    
    counts = counts +1  # Removes divide by zero if registers are not measured
    
    # if 0 in counts:
    #     counts = counts +1
    
    n_measurements = np.sum(counts)
    probs = counts/n_measurements # Convert measurements to probabilities
    
    regprob = np.zeros(2**nr) # Initialize array of register probabilities 
    for a_ in range(2**na): # Obtain register probabilities
        regprob = regprob + probs[a_::nstates_anc]
        
    proj = list([]) # Initialize array of ancilla*register probabilities

    for anc_states in range(2**na): # Obtain ancilla*register probabilities
        proj_list = probs[anc_states::nstates_anc]
        proj.append(proj_list)

    # regprob should sum to 1
    # All columns in anc_probs should sum to 1
    # Returns [regprob1, regprob2,..]
    return np.array(regprob), np.array(proj) # Register probabilities, ancilla*register probabilities

def gen_sequences(N): # Slightly quicker way of producing the bitmap_t matrix, but still not usable for n_a = 32
    # Replaces bitmap_t = np.array([np.array(bitstr) for bitstr in itertools.product([0,1],repeat=n_ancilla)]).T
    sequences = []
    for m in reversed(range(N)):
        power = 2**m
        sequence = ([0] * power + [1] * power) * (2**N//(2*2**m))
        sequences.append(sequence)
    return np.array(sequences)


def cfgrad_vec(angles, matrix, na,nr, n_measurements, bitmap, param_list): # Calculates gradient of cost function
    # Returns 2 outputs
    vvec = copy.deepcopy(angles)
    
    nq, cdepth, nparam = na+nr, int(len(angles)/(na+nr)), len(angles) # Get number of qubits, depth of circuit number of parameters

    mes_results = cdict(circuit(nq, cdepth, vvec),n_measurements)
    input_angles_measurement = mes_results
    proj_, proj_1 = getproj(mes_results,na,nr) # P(theta), proj_1/proj is the probability of the ancillas
    
#     bitmap_t = gen_sequences(na)
    
    pmat = list([])
    for reg_index in range(2**nr): # probability matrix (at theta) of ancilla
        ancprobvec = list([])
        for anc_index in range(na):
            ancprobvec.append(np.sum((proj_1/proj_).T[reg_index][np.where(bitmap[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)
    
    gradient_vec = np.zeros(nparam)
    
    for theta in param_list:  # Anything that requires _p or _m or dproj has to be inside this loop
        vvec_p = copy.deepcopy(angles) 
        vvec_m = copy.deepcopy(angles) 
        vvec_p[theta] = vvec_p[theta] + np.pi/2 # theta + pi/2
        vvec_m[theta] = vvec_m[theta] - np.pi/2 # theta - pi/2

        serkstate_p = cdict(circuit(nq, cdepth, vvec_p),n_measurements)
        serkstate_m = cdict(circuit(nq, cdepth, vvec_m),n_measurements)

        proj_p, proj_1p = getproj(serkstate_p,na,nr) # P(theta + pi/2)
        proj_m, proj_1m = getproj(serkstate_m,na,nr) # P(theta - pi/2)

        dproj_1 = 0.5*(proj_1p - proj_1m) # Parameter shift for ancilla 
        dproj_ = 0.5*(proj_p - proj_m) # Parameter shift for registers

        # anc_d_prob = (dproj_1/proj_) - (proj_1/(proj_)**2)*dproj_
        dprob = np.multiply(1/proj_, dproj_1) - np.multiply(np.multiply(np.multiply(1/proj_,1/proj_),proj_1), dproj_)
        

        dmat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(dprob.T[reg_index][np.where(bitmap[anc_index] == 1)]))
            dmat.append(dprobvec)
        dmat = np.array(dmat)
    
        diag_subgrad = 0
        offdiag_subgrad = 0

        for subsystem_index1 in range(2**nr): # loop over subsystems
            for subsystem_index2 in range(subsystem_index1, 2**nr):

                # Gets the submatrices along the diagonal and off diagonal
                submat = matrix[subsystem_index1*na:subsystem_index1*(na)+na, subsystem_index2*na:subsystem_index2*(na)+na]

                if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                    bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla

                    for item in itertools.product([0,1],repeat=na): # Treat it as complete encoding
                        subsystem_bitstring = np.array(item) # somehow faster to generate bitstrings than to transpose bitmap

                        # Essentially calculating C_subsystem = <xTAx>
                        diag_subgrad = diag_subgrad + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*dprob[bitstr_index][subsystem_index1]
                        bitstr_index = bitstr_index + 1


                else: # Create pmat to obtain probabilities of off-diagonal subsystems
                    pdmat = np.outer(pmat[subsystem_index1],dmat[subsystem_index2]) + np.outer(dmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                    offdiag_subgrad = offdiag_subgrad + np.sum(np.multiply(pdmat,submat)) # calculates cost function using hadamard product

        gradient_vec[theta]= diag_subgrad+2*offdiag_subgrad # Multiply by 2 since calculations were done for only upper diagonal matrix
    
    return gradient_vec, input_angles_measurement


def anc_cf(measurement_results, matrix, n_ancilla, bitmap):    
    
    n_reg = int(np.ceil(np.log2(len(matrix)/n_ancilla))) # Obtains number of register qubits

    reg_prob, reg_anc_prob = getproj(measurement_results,n_ancilla,n_reg) # Get register and ancilla probabilities from input state
    anc_prob = reg_anc_prob/reg_prob
    
    # Transposed List of all possible bitstrings from ancilla qubits. Used to construct pmat. 
    ################################################################
    ############## WiLL be long if n_a is large (>16) ##############
    ################################################################

    
    # Pmat is a rectangular matrix. For 2anc, each row is [c^2 + d^2, b^2 + d^2]
    # The same pmat is used for all subsystems so it only has to be generated once
    pmat = list([]) # Creates the matrix of probabilities for the bits to be 1.
    
    for reg_index in range(2**n_reg):
        ancprobvec = list([])
        for anc_index in range(n_ancilla):
            ancprobvec.append(np.sum(anc_prob.T[reg_index][np.where(bitmap[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)
    
    diag_subcf = 0
    offdiag_subcf = 0


    for subsystem_index1 in range(2**n_reg): # loop over subsystems
        for subsystem_index2 in range(subsystem_index1,2**n_reg):

            # Gets the submatrices along the diagonal and off diagonal
            submat = matrix[subsystem_index1*n_ancilla:subsystem_index1*(n_ancilla)+n_ancilla, subsystem_index2*n_ancilla:subsystem_index2*(n_ancilla)+n_ancilla]

            if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla
                
                for item in itertools.product([0,1],repeat=n_ancilla): # Treat it as complete encoding
                    subsystem_bitstring = np.array(item) 
                    
                    # Essentially calculating C_subsystem = <xTAx>
                    diag_subcf = diag_subcf + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*anc_prob[bitstr_index][subsystem_index1]
                    bitstr_index = bitstr_index + 1

            else: # Create pmat to obtain probabilities of off-diagonal subsystems
                sub_pmat = np.outer(pmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                offdiag_subcf = offdiag_subcf + np.sum(np.multiply(sub_pmat,submat)) # calculates cost function using hadamard product

    offdiag_subcf = offdiag_subcf*2 # Multiply by 2 since calculations were done for only upper diagonal matrix
    return diag_subcf+offdiag_subcf # Sum diagonal and off diagonal terms


def fullenc_cf(measurement_results,matrix):
    
    counts = np.zeros(2**len(matrix))
    
    for key in measurement_results:
        pos = int(key,2)
        counts[pos] = measurement_results[key] 
    
    # counts = counts +1 # Should not be needed for full encoding
    
    n_measurements = np.sum(counts)
    probs = counts/n_measurements
    
    cf = 0
    
    for item in itertools.product([0,1],repeat=len(matrix)):
        bitstr = np.array(item)

        bitstr_int = ''.join(str(bit) for bit in item)
        cf = cf+ (bitstr.dot(matrix)).dot(bitstr) * probs[int(bitstr_int,2)]
        
    return cf

def calcf(measurement_results,matrix, n_ancilla,bitmap):
    
    if len(matrix) != n_ancilla:
        return anc_cf(measurement_results, matrix, n_ancilla, bitmap)
    
    else: 
        return fullenc_cf(measurement_results,matrix)

def solvequbo(circuit,n_bitstr, matrix, na, n_measurements):

    np.random.seed()
    mes_results = cdict(circuit, n_measurements)
    nr = int(np.ceil(np.log2(len(matrix)/na)))
    
    cprob, all_prob = getproj(mes_results,na,nr)
    ancprob = all_prob/cprob # Takes out only the probabilities of the 1-state
    
    bitstr_prob = np.transpose(ancprob)
    
    subsystemlist = list([])
    for item in itertools.product([0,1],repeat=na):
        subsystemlist.append(list(item))
        
    qubo_cf = list([])
    for k in range(n_bitstr):
        
        bitstring = np.array([subsystemlist[np.random.choice(np.arange(2**na), p = bitprob)] for bitprob in bitstr_prob]).flatten()
        qubocf = (bitstring.dot(matrix)).dot(bitstring)
        qubo_cf.append(qubocf)
    
    
    return np.array(qubo_cf)


########################################################################
##################### Infinite measurement stuff #######################
########################################################################

def statevec(circuit):
    copyserk = copy.deepcopy(circuit) # To prevent changing the actual circuit
    copyserk.remove_final_measurements()
    job = execute(copyserk,Aer.get_backend('statevector_simulator')).result()
    stvec = job.get_statevector()
    return stvec

def getproj_inf(state,na,nr):
    
    nstates_anc = 2**na
    
    probs = np.abs(state)**2 # Convert coefficients to probabilities
    regprob = np.zeros(2**nr) # Initialize array of register probabilities
    
    for a_ in range(2**na): # Obtain register probabilities
        regprob = regprob + probs[a_::nstates_anc]
        
    proj = list([]) # Initialize array of ancilla*register probabilities

    for anc_states in range(2**na): # Obtain ancilla*register probabilities
        proj_list = probs[anc_states::nstates_anc]
        proj.append(proj_list)

    # regprob should sum to 1
    # All columns in anc_probs should sum to 1
    # Returns [regprob1, regprob2,..]
    return np.array(regprob), np.array(proj)

def anc_cf_inf(state, matrix, n_ancilla,bitmap):
        
    n_reg = int(np.ceil(np.log2(len(matrix)/n_ancilla))) # Obtains number of register qubits

    reg_prob, reg_anc_prob = getproj_inf(state,n_ancilla,n_reg) # Get register and ancilla probabilities from input state
    anc_prob = reg_anc_prob/reg_prob
    
    # Transposed List of all possible bitstrings from ancilla qubits. Used to construct pmat. 
    ################################################################
    ############## WiLL be long if n_a is large (>16) ##############
    ################################################################
#     bitmap_t = np.array([np.array(bitstr) for bitstr in itertools.product([0,1],repeat=n_ancilla)]).T
    # bitmap_t = gen_sequences(n_ancilla) # Faster way of generating the bitmap matrix. Original code above

    
    # Pmat is a rectangular matrix. For 2anc, each row is [c^2 + d^2, b^2 + d^2]
    # The same pmat is used for all subsystems so it only has to be generated once
    pmat = list([]) # Creates the matrix of probabilities for the bits to be 1.
    
    for reg_index in range(2**n_reg):
        ancprobvec = list([])
        for anc_index in range(n_ancilla):
            ancprobvec.append(np.sum(anc_prob.T[reg_index][np.where(bitmap[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)
    
    diag_subcf = 0
    offdiag_subcf = 0

    for subsystem_index1 in range(2**n_reg): # loop over subsystems
        for subsystem_index2 in range(subsystem_index1,2**n_reg):

            # Gets the submatrices along the diagonal and off diagonal
            submat = matrix[subsystem_index1*n_ancilla:subsystem_index1*(n_ancilla)+n_ancilla, subsystem_index2*n_ancilla:subsystem_index2*(n_ancilla)+n_ancilla]

            if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla
                
                for item in itertools.product([0,1],repeat=n_ancilla): # Treat it as complete encoding
                    subsystem_bitstring = np.array(item) 
                    
                    # Essentially calculating C_subsystem = <xTAx>
                    diag_subcf = diag_subcf + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*anc_prob[bitstr_index][subsystem_index1]
                    bitstr_index = bitstr_index + 1

            else: # Create pmat to obtain probabilities of off-diagonal subsystems
                sub_pmat = np.outer(pmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                offdiag_subcf = offdiag_subcf + np.sum(np.multiply(sub_pmat,submat)) # calculates cost function using hadamard product

    offdiag_subcf = offdiag_subcf*2 # Multiply by 2 since calculations were done for only upper diagonal matrix
    return diag_subcf+offdiag_subcf


def cfgrad_vec_inf(angles, matrix, na,nr, bitmap, param_list): # Calculates gradient of cost function
    
    nq, cdepth, nparam = na+nr, int(len(angles)/(na+nr)), len(angles) # Get number of qubits, depth of circuit number of parameters

    mes_results = statevec(circuit(nq, cdepth, angles))
    # state_at_theta = mes_results
    proj_, proj_1 = getproj_inf(mes_results,na,nr) # P(theta), proj_1/proj is the probability of the ancillas
    
#     bitmap_t = gen_sequences(na)
    
    pmat = list([])
    for reg_index in range(2**nr): # probability matrix (at theta) of ancilla
        ancprobvec = list([])
        for anc_index in range(na):
            ancprobvec.append(np.sum((proj_1/proj_).T[reg_index][np.where(bitmap[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)
    
    gradient_vec_inf = np.zeros(nparam)
    
    for theta in param_list:  # Anything that requires _p or _m or dproj has to be inside this loop
        vvec_p = angles.copy()
        vvec_m = angles.copy()
        vvec_p[theta] = vvec_p[theta] + np.pi/2 # theta + pi/2
        vvec_m[theta] = vvec_m[theta] - np.pi/2 # theta - pi/2

        serkstate_p = statevec(circuit(nq, cdepth, vvec_p))
        serkstate_m = statevec(circuit(nq, cdepth, vvec_m))

        proj_p, proj_1p = getproj_inf(serkstate_p,na,nr) # P(theta + pi/2)
        proj_m, proj_1m = getproj_inf(serkstate_m,na,nr) # P(theta - pi/2)

        dproj_1 = 0.5*(proj_1p - proj_1m) # Parameter shift for ancilla 
        dproj_ = 0.5*(proj_p - proj_m) # Parameter shift for registers

        # anc_d_prob = (dproj_1/proj_) - (proj_1/(proj_)**2)*dproj_
        dprob = np.multiply(1/proj_, dproj_1) - np.multiply(np.multiply(np.multiply(1/proj_,1/proj_),proj_1), dproj_)
        

        dmat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(dprob.T[reg_index][np.where(bitmap[anc_index] == 1)]))
            dmat.append(dprobvec)
        dmat = np.array(dmat)
        
    
        diag_subgrad = 0
        offdiag_subgrad = 0

        for subsystem_index1 in range(2**nr): # loop over subsystems
            for subsystem_index2 in range(subsystem_index1, 2**nr):

                # Gets the submatrices along the diagonal and off diagonal
                submat = matrix[subsystem_index1*na:subsystem_index1*(na)+na, subsystem_index2*na:subsystem_index2*(na)+na]

                if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                    bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla

                    for item in itertools.product([0,1],repeat=na): # Treat it as complete encoding
                        subsystem_bitstring = np.array(item) # somehow faster to generate bitstrings than to transpose bitmap

                        # Essentially calculating C_subsystem = <xTAx>
                        diag_subgrad = diag_subgrad + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*dprob[bitstr_index][subsystem_index1]
                        bitstr_index = bitstr_index + 1


                else: # Create pmat to obtain probabilities of off-diagonal subsystems
                    pdmat = np.outer(pmat[subsystem_index1],dmat[subsystem_index2]) + np.outer(dmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                    offdiag_subgrad = offdiag_subgrad + np.sum(np.multiply(pdmat,submat)) # calculates cost function using hadamard product

        gradient_vec_inf[theta]= diag_subgrad+2*offdiag_subgrad # Multiply by 2 since calculations were done for only upper diagonal matrix
    
    return gradient_vec_inf

def fullenc_cf_inf(state,matrix):
    probs = np.abs(state)**2
    
    cf = 0
    
    for item in itertools.product([0,1],repeat=len(matrix)):
        bitstr = np.array(item)

        bitstr_int = ''.join(str(bit) for bit in item)
        cf = cf+ (bitstr.dot(matrix)).dot(bitstr) * probs[int(bitstr_int,2)]
        
    return cf

def calcf_inf(state,matrix, n_ancilla,bitmap):
    
    if len(matrix) != n_ancilla:
        return anc_cf_inf(state, matrix, n_ancilla, bitmap)
    
    else: 
        return fullenc_cf_inf(state,matrix)
    
    
def solvequbo_inf(circuit,n_bitstr, matrix, na):

    np.random.seed()
    state_from_circuit = statevec(circuit)
    nr = int(np.ceil(np.log2(len(matrix)/na)))
    
    cprob, all_prob = getproj_inf(state_from_circuit,na,nr)
    ancprob = all_prob/cprob # Takes out only the probabilities of the 1-state
    
    bitstr_prob = np.transpose(ancprob)
    
    subsystemlist = list([])
    for item in itertools.product([0,1],repeat=na):
        subsystemlist.append(list(item))
        
    qubo_cf = list([])
    for k in range(n_bitstr):
        
        bitstring = np.array([subsystemlist[np.random.choice(np.arange(2**na), p = bitprob)] for bitprob in bitstr_prob]).flatten()
        qubocf = (bitstring.dot(matrix)).dot(bitstring)
        qubo_cf.append(qubocf)
    
    
    return np.array(qubo_cf)
    
########################################################################
########################################################################
########################################################################



class SGD_Na:
        
    def __init__(self, learning_rate,n_ancilla, max_iterations = 1000, eps = 1e-5, ntrain = 'all'): # learning rate and max iter as init attributes
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.n_ancilla = n_ancilla
        self.eps = np.float(eps)
        self.ntrain = ntrain
        
        if n_ancilla <= 16:
            pass
        else:
            print("MORE THAN 16 ANCILLAS. Might take too long to calculate CF in each subsystem")
            
        self.bitmap_t = gen_sequences(self.n_ancilla)
        
    def minimize(self, n_qubits, c_depth, matrix, angles, measurements, max_shots = None):
        
        print("Stochastic gradient descent")
                
        sst1 = time.time()
        
        # Initial params/data/hyperparams
        nreg = int(np.ceil(np.log2(len(matrix)/self.n_ancilla)))        
        initial_angles = copy.deepcopy(angles)
        nparam = len(initial_angles)
        # init_cf = calcf(cdict(circuit(n_qubits,c_depth,initial_angles),measurements),matrix, self.n_ancilla,self.bitmap_t)
        LR = self.learning_rate    
        

        # For saving data
        cflist = list([])
        # cf_inf_list, x_inf_plot = list([]),list([])
        anglist = list([])
        # cflist.append(init_cf)
        anglist.append(initial_angles)
        
        
        # Gradient of cost function wrt to each angle
        graddict = list([])
        for thet in range(nparam):
            graddict.append(list([]))

        
        nparam_list = np.array([tt for tt in range(nparam)])

        for curr_iter in range(self.max_iterations):       
            
            if self.ntrain == 'all':
                pass
            else: 
                nparam_list = np.random.choice(nparam, np.random.randint(1,nparam), replace=False)
            

            dc_dtheta, measure_once = cfgrad_vec(initial_angles, matrix, self.n_ancilla,nreg, measurements,self.bitmap_t,nparam_list)
            cf_before_SGD = calcf(measure_once,matrix, self.n_ancilla, self.bitmap_t) # CF before each iteration

            for theta in range(len(dc_dtheta)): 
                graddict[theta].append(dc_dtheta[theta]) # Gradient of cost function wrt to each angle
                
            initial_angles = initial_angles - LR* dc_dtheta # gradient descent. Updates all angles at once
            
#             print((cf- matmin)/(matmax - matmin)) # <------ for verbose mode
            
            cflist.append(cf_before_SGD)
            anglist.append(list(initial_angles))
            
            if curr_iter > 1 and curr_iter%100 == 0:
#                 LR = LR*0.85
                print("Completed iter:", curr_iter, ". Current learning rate:", LR)
            elif curr_iter == 0:
                print("SGD Initial CF:",round(cflist[0],5), ". Initial learning rate:", LR, "Max iterations:", self.max_iterations)
                
            
        # This is done at the end cause params get saved first, then CF. 
        # So after all the iter, cf list is lacking the last cf entry.
        cflist.append(calcf(cdict(circuit(n_qubits,c_depth,initial_angles),measurements),matrix, self.n_ancilla,self.bitmap_t))


        anglist.append(anglist[np.argmin(cflist)])
        anglist = np.array(anglist)
        cflist.append(np.min(cflist))

            
        sst2 = time.time()
        self.time = round(sst2 - sst1,5)
        self.mincf = round(cflist[-2],5)
        self.optvec = anglist[-1]%(2*np.pi)
        self.optev = curr_iter
        self.cflist = cflist
        self.vectlist = anglist%(2*np.pi)
        self.graddict = graddict # Gradient of cost function wrt to each angle
        
        return anglist[-1], cflist[-1]
    
    
    
    ################################################################################################
    #++++++++++++++++++++++++++++++++++++++ ADAM QUANTUM OPT +++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    def adam(self, n_qubits, c_depth, matrix, angles, measurements, max_shots = None):
               
        print("ADAM optimizer")
        
        sst1 = time.time()
        
        # Initial params/data/hyperparams

        nreg = int(np.ceil(np.log2(len(matrix)/self.n_ancilla)))

        initial_angles = copy.deepcopy(angles)
        nparam = len(initial_angles)
        # init_cf = calcf(cdict(circuit(n_qubits,c_depth,initial_angles),measurements),matrix, self.n_ancilla,self.bitmap_t)
           
        # Hyperparameters for ADAM
        alpha = 0.002 # Default val: 0.001
        b1 = 0.9 # Default val: 0.9
        b2 = 0.999 # Default val: 0.999
        eps = int(1e-8) # To prevent divide by zero
        
        # Initialize moments as 0-vectors
        mvec = np.zeros(nparam) 
        vtvec = np.zeros(nparam)

        # For saving data
        cflist = list([])
        anglist = list([])
        cf_inf_list, x_inf_plot = list([]),list([])
        # cflist.append(init_cf)
        anglist.append(initial_angles)
        
        # Gradient of cost function wrt to each angle
        graddict = list([])
        for thet in range(nparam):
            graddict.append(list([]))
            
        if max_shots == None: # Used for stopping criteria once max number of shots is used up
            pass
        else:
            print("++++ MAX SHOTS NOT WORKING FOR NOW BECAUSE OF BATCH TRAINING ++++")
        #     shots_per_eval = ((2*nparam)+1) * measurements
        #     self.max_iterations = np.min([self.max_iterations , np.int(np.floor(max_shots/shots_per_eval))])
        #     print("Max iterations limited by number of shots:", self.max_iterations)

        nparam_list = np.array([tt for tt in range(nparam)])
        
            
        for curr_iter in range(1,self.max_iterations+1):      
            
            if self.ntrain == 'all':
                pass
            else: 
                nparam_list = np.random.choice(nparam, np.random.randint(1,nparam), replace=False)
            
            dc_dtheta, measure_once = cfgrad_vec(initial_angles, matrix, self.n_ancilla,nreg, measurements,self.bitmap_t, nparam_list)

            for theta in range(len(dc_dtheta)): 
                graddict[theta].append(dc_dtheta[theta]) # Gradient of cost function wrt to each angle
                
            ##### Steps for ADAM optimizer ####                
            test_grad = copy.deepcopy(dc_dtheta)
            mvec = b1*mvec + (1-b1)*test_grad # Update first moment
            vtvec = b2*vtvec + (1-b2)*(test_grad**2) # Update second moment
            
            mhat = mvec/(1-(b1**curr_iter)) # Bias correction for first moment
            vhat = vtvec/(1-(b2**curr_iter)) # Bias correction for second moment
            #################################
                        
            initial_angles = initial_angles - (alpha*mhat)/(np.sqrt(vhat) + eps) # gradient descent. Updates all angles at once
            cf_before_SGD = calcf(measure_once,matrix, self.n_ancilla, self.bitmap_t) # CF after each iteration
#             print((cf- matmin)/(matmax - matmin)) # <------ for verbose mode
            
            cflist.append(cf_before_SGD)
            anglist.append(list(initial_angles))
        
            
            if curr_iter > 1 and curr_iter%100 == 0:
#                 LR = LR*0.85
                print("Current iter:", curr_iter, ", Current CF:", round(cflist[-1],4))
            elif curr_iter == 0:
                print("ADAM Optimizer. Initial CF:",round(cflist[0],5), ". Initial learning rate:", alpha)   


        # +++++++++ ADAM QUANTUM OPT +++++++++
        cflist.append(calcf(cdict(circuit(n_qubits,c_depth,initial_angles),measurements),matrix, self.n_ancilla,self.bitmap_t))
        
        anglist.append(anglist[np.argmin(cflist)])        
        cflist.append(np.min(cflist))
        
            
        sst2 = time.time()
        self.time = round(sst2 - sst1,5)
        self.mincf = round(cflist[-2],5)
        # self.mincf_inf = round(finalcf_inf,5)
        self.optvec = anglist[-1]
        self.optev = curr_iter
        self.cflist = cflist
        self.cf_inf_plot = [x_inf_plot,cf_inf_list]
        self.vectlist = anglist
        self.graddict = graddict # Gradient of cost function wrt to each angle
        
        return anglist[-1], cflist[-1]
    
    
    
############################################################
###################### HESSIAN MATRIX ######################
############################################################
        
def hess_elem(angles, matrix, na,nr, nq, cdepth , thet_i,thet_j, bitmap_t):
    ######## number of ancilla qubits, number of register qubits, total qubits, depth of circuit
    
    vvec = copy.deepcopy(angles)
    # Get number of qubits, depth of circuit number of parameters

    state = statevec(circuit(nq, cdepth, vvec))
    proj_, proj_1 = getproj_inf(state,na,nr) # P(theta), proj_1/proj is the probability of the ancillas

    pmat = list([])
    for reg_index in range(2**nr): # probability matrix (at theta) of ancilla
        ancprobvec = list([])
        for anc_index in range(na):
            ancprobvec.append(np.sum((proj_1/proj_).T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)
       
    if thet_i == thet_j: # need to calculate double derivative using parameter shift. 
        
        ########################## Parameter shifts ##########################
        # For first derivatives
        vvec_p = copy.deepcopy(angles) 
        vvec_m = copy.deepcopy(angles) 
        vvec_p[thet_i] = vvec_p[thet_i] + np.pi/2 # theta + pi
        vvec_m[thet_i] = vvec_m[thet_i] - np.pi/2 # theta - pi
        
        serkstate_p = statevec(circuit(nq, cdepth, vvec_p))
        serkstate_m = statevec(circuit(nq, cdepth, vvec_m))

        proj_p, proj_1p = getproj_inf(serkstate_p,na,nr) # P(theta + pi/2)
        proj_m, proj_1m = getproj_inf(serkstate_m,na,nr) # P(theta - pi/2)

        dproj_1 = 0.5*(proj_1p - proj_1m) # Double Parameter shift for ancilla 
        dproj_ = 0.5*(proj_p - proj_m) # Double Parameter shift for register
        
        # For second derivatives
        vvec_p = copy.deepcopy(angles) 
        vvec_m = copy.deepcopy(angles) 
        vvec_p[thet_i] = vvec_p[thet_i] + np.pi # theta + pi
        vvec_m[thet_i] = vvec_m[thet_i] - np.pi # theta - pi
        
        serkstate_p = statevec(circuit(nq, cdepth, vvec_p))
        serkstate_m = statevec(circuit(nq, cdepth, vvec_m))

        proj_p, proj_1p = getproj_inf(serkstate_p,na,nr) # P(theta + pi)
        proj_m, proj_1m = getproj_inf(serkstate_m,na,nr) # P(theta - pi)

        d2proj_1 = 0.25*(proj_1p + proj_1m - 2*proj_1) # Double Parameter shift for ancilla 
        d2proj_ = 0.25*(proj_p + proj_m - 2*proj_) # Double Parameter shift for register
        
        ########################## Derivative matrices ##########################
    
        dprob = np.multiply(1/proj_, dproj_1) - np.multiply(np.multiply(np.multiply(1/proj_,1/proj_),proj_1), dproj_)
        d2prob = (1/(proj_)**3)*(proj_1*(2*(dproj_**2) - proj_*d2proj_ )) + proj_*(-2*dproj_ * dproj_1 + proj_*d2proj_1)
         
        dmat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(dprob.T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
            dmat.append(dprobvec)
        dmat = np.array(dmat)
        
        d2mat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(d2prob.T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
            d2mat.append(dprobvec)
        d2mat = np.array(d2mat)
        
        ########################## Calculating derivatives ##########################
        
        ## (d_i)(d_i )f1 * f2 + 2*(d_i)f1 (d_i)f2 + (d_i)(d_i)f2 * f1
        diag_subgrad = 0
        offdiag_subgrad = 0

        for subsystem_index1 in range(2**nr): # loop over subsystems
            for subsystem_index2 in range(subsystem_index1, 2**nr):

                # Gets the submatrices along the diagonal and off diagonal
                submat = matrix[subsystem_index1*na:subsystem_index1*(na)+na, subsystem_index2*na:subsystem_index2*(na)+na]

                if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                    bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla

                    for item in itertools.product([0,1],repeat=na): # Treat it as complete encoding
                        subsystem_bitstring = np.array(item) # somehow faster to generate bitstrings than to transpose bitmap

                        # Essentially calculating C_subsystem = <xTAx>
                        diag_subgrad = diag_subgrad + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*d2prob[bitstr_index][subsystem_index1]
                        bitstr_index = bitstr_index + 1


                else: # Create pmat to obtain probabilities of off-diagonal subsystems
                    pdmat = np.outer(pmat[subsystem_index1],d2mat[subsystem_index2]) + np.outer(d2mat[subsystem_index1],pmat[subsystem_index2]) + 2*np.outer(dmat[subsystem_index1],dmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                    offdiag_subgrad = offdiag_subgrad + np.sum(np.multiply(pdmat,submat)) # calculates cost function using hadamard product

        hessian_element= diag_subgrad+2*offdiag_subgrad # Multiply by 2 since calculations were done for only upper diagonal matrix
    
    else:
        
        ########################## Parameter shifts ##########################
        # W.r.t. theta_i
        vvec_p = copy.deepcopy(angles) 
        vvec_m = copy.deepcopy(angles) 
        vvec_p[thet_i] = vvec_p[thet_i] + np.pi/2 # theta + pi
        vvec_m[thet_i] = vvec_m[thet_i] - np.pi/2 # theta - pi
        
        serkstate_p = statevec(circuit(nq, cdepth, vvec_p))
        serkstate_m = statevec(circuit(nq, cdepth, vvec_m))

        proj_p, proj_1p = getproj_inf(serkstate_p,na,nr) # P(theta + pi/2)
        proj_m, proj_1m = getproj_inf(serkstate_m,na,nr) # P(theta - pi/2)

        diproj_1 = 0.5*(proj_1p - proj_1m) # Double Parameter shift for ancilla 
        diproj_ = 0.5*(proj_p - proj_m) # Double Parameter shift for register
        
        # W.r.t. theta_j
        vvec_p = copy.deepcopy(angles) 
        vvec_m = copy.deepcopy(angles) 
        vvec_p[thet_j] = vvec_p[thet_j] + np.pi/2 # theta + pi
        vvec_m[thet_j] = vvec_m[thet_j] - np.pi/2 # theta - pi
        
        serkstate_p = statevec(circuit(nq, cdepth, vvec_p))
        serkstate_m = statevec(circuit(nq, cdepth, vvec_m))

        proj_p, proj_1p = getproj_inf(serkstate_p,na,nr) # P(theta + pi/2)
        proj_m, proj_1m = getproj_inf(serkstate_m,na,nr) # P(theta - pi/2)

        djproj_1 = 0.5*(proj_1p - proj_1m) # Double Parameter shift for ancilla 
        djproj_ = 0.5*(proj_p - proj_m) # Double Parameter shift for register
        
        # W.r.t. theta_ij
        vvec_pp = copy.deepcopy(angles) 
        vvec_pm = copy.deepcopy(angles) 
        vvec_mp = copy.deepcopy(angles) 
        vvec_mm = copy.deepcopy(angles) 
        
        vvec_pp[thet_i],vvec_pp[thet_j] = vvec_pp[thet_i] + np.pi/2, vvec_pp[thet_j] + np.pi/2 # theta_i + pi/2, theta_j + pi/2
        vvec_pm[thet_i],vvec_pm[thet_j] = vvec_pm[thet_i] + np.pi/2, vvec_pm[thet_j] - np.pi/2 # theta_i + pi/2, theta_j - pi/2
        vvec_mp[thet_i],vvec_mp[thet_j] = vvec_mp[thet_i] - np.pi/2, vvec_mp[thet_j] + np.pi/2 # theta_i - pi/2, theta_j + pi/2
        vvec_mm[thet_i],vvec_mm[thet_j] = vvec_mm[thet_i] - np.pi/2, vvec_mm[thet_j] - np.pi/2 # theta_i - pi/2, theta_j - pi/2
        
        serkstate_pp = statevec(circuit(nq, cdepth, vvec_pp))
        serkstate_pm = statevec(circuit(nq, cdepth, vvec_pm))
        serkstate_mp = statevec(circuit(nq, cdepth, vvec_mp))
        serkstate_mm = statevec(circuit(nq, cdepth, vvec_mm))

        proj_pp, proj_1pp = getproj_inf(serkstate_pp,na,nr) # P(theta + pi/2)
        proj_pm, proj_1pm = getproj_inf(serkstate_pm,na,nr) # P(theta - pi/2)
        proj_mp, proj_1mp = getproj_inf(serkstate_mp,na,nr) # P(theta + pi/2)
        proj_mm, proj_1mm = getproj_inf(serkstate_mm,na,nr) # P(theta - pi/2)

        didjproj_1 = 0.25*(proj_1pp - proj_1pm - proj_1mp + proj_1mm) # Double Parameter shift for ancilla 
        didjproj_ = 0.25*(proj_pp - proj_pm - proj_mp + proj_mm) # Double Parameter shift for register
        
        ########################## Derivative matrices ##########################
        
        diprob = np.multiply(1/proj_, diproj_1) - np.multiply(np.multiply(np.multiply(1/proj_,1/proj_),proj_1), diproj_)
        djprob = np.multiply(1/proj_, djproj_1) - np.multiply(np.multiply(np.multiply(1/proj_,1/proj_),proj_1), djproj_)
        
        didjprob = (1/(proj_)**3)*( proj_1 * (2* djproj_ * diproj_ - proj_ * didjproj_) 
                    + proj_*(-djproj_1 * diproj_ - djproj_*diproj_1 + proj_*didjproj_1) )
        
        dimat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(diprob.T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
            dimat.append(dprobvec)
        dimat = np.array(dimat)
        
        djmat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(djprob.T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
            djmat.append(dprobvec)
        djmat = np.array(djmat)
        
        didjmat = list([])
        for reg_index in range(2**nr):
            dprobvec = list([])
            for anc_index in range(na):
                dprobvec.append(np.sum(didjprob.T[reg_index][np.where(bitmap_t[anc_index] == 1)]))
            didjmat.append(dprobvec)
        didjmat = np.array(didjmat)
        
        ########################## Calculating derivatives ##########################
        
        ## (d_i)(d_j )f1 * f2 + (d_i)f1 (d_j)f2 + (d_i)f2 (d_j)f1 + (d_i)(d_j )f2 * f1
        diag_subgrad = 0
        offdiag_subgrad = 0
        
        for subsystem_index1 in range(2**nr): # loop over subsystems
            for subsystem_index2 in range(subsystem_index1, 2**nr):

                # Gets the submatrices along the diagonal and off diagonal
                submat = matrix[subsystem_index1*na:subsystem_index1*(na)+na, subsystem_index2*na:subsystem_index2*(na)+na]

                if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                    bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla

                    for item in itertools.product([0,1],repeat=na): # Treat it as complete encoding
                        subsystem_bitstring = np.array(item) # somehow faster to generate bitstrings than to transpose bitmap

                        # Essentially calculating C_subsystem = <xTAx>
                        diag_subgrad = diag_subgrad + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*didjprob[bitstr_index][subsystem_index1]
                        bitstr_index = bitstr_index + 1


                else: # Create pmat to obtain probabilities of off-diagonal subsystems
                    pdmat = (np.outer(didjmat[subsystem_index1],pmat[subsystem_index2]) 
                            + np.outer(dimat[subsystem_index1],djmat[subsystem_index2]) 
                            + np.outer(djmat[subsystem_index1],dimat[subsystem_index2]) 
                            + np.outer(pmat[subsystem_index1],didjmat[subsystem_index2])) # Gets the correct probabilities for the subsystems
                            
                    offdiag_subgrad = offdiag_subgrad + np.sum(np.multiply(pdmat,submat)) # calculates cost function using hadamard product

        hessian_element= diag_subgrad+2*offdiag_subgrad # Multiply by 2 since calculations were done for only upper diagonal matrix
    
    return hessian_element


def hessmat(angles, matrix, na):
    
    
    nr = np.int(np.ceil(np.log2(len(matrix)/na)))
    nq = nr + na
    
    cdepth  = int(np.floor(len(angles)/(na+nr)))
    angles = angles[:nq*cdepth]
    nparam = len(angles)
    
    hessian_matrix = np.zeros((nparam, nparam))
    bitmapT = gen_sequences(na)
    
    print("Calculating Hessian. Anc:",na, ", Nq:",nq, ", layers:",cdepth, ", Hessian matrix size:", str(nparam)+"x"+str(nparam))
    
    #### Hessian matrix should be symmetric ####
    
    for i in range(nparam):
        for j in range(i,nparam):
            matrix_element = hess_elem(angles, matrix, na,nr, nq, cdepth, i,j, bitmapT)
            hessian_matrix[i][j] = np.round(matrix_element,7)
            
    for i in range(nparam):
        for j in range(i,nparam): 
            hessian_matrix[j][i] = hessian_matrix[i][j]
            
    return hessian_matrix



#########################################################################################################
# Classical optimization of QUBO for general ancilla using SLSQP. 
# NOT PERFECT. Probs can still take small negative values or be more than 1 (but get renormalized anyway)
#########################################################################################################

def get_1_more(numbers): # Method to get n+1 numbers from n numbers. 
    # Input is an array of elements between 0 and 1
    # Output is an array of n+1 elements between 0 and 1. Output array sums to 1
    
    num_arr = np.zeros(len(numbers) + 1)
    
    for i in range(len(numbers)):
        num_arr[i] = np.prod(numbers[:i])*(1-numbers[i])
        
    num_arr[-1] = np.prod(numbers)
    return num_arr



def cf_classical(params,matrix,n_anc,bitmap):
    

    n_reg = int(np.ceil(np.log2(len(matrix)/n_anc)))
    n_subgroups = int((len(matrix)/n_anc))
    
    
    if len(params) == ((2**n_anc)-1)*n_subgroups: # If using less parameters to get probabilities
        prob_mat = list([])
        anc_param = (2**n_anc)-1
        
        for i_ in range(n_subgroups):
            anc_numbers = params[i_*anc_param:i_*anc_param+anc_param] # makes it positive (-ve numbers are small)
            anc_probs = get_1_more(anc_numbers)
            prob_mat.append(anc_probs)
    
    elif len(params) == ((2**n_anc))*n_subgroups: # If using more parameters to get probabilities
        prob_mat = list([])       
        anc_param = (2**n_anc)    
        
        for i_ in range(n_subgroups):        
            anc_probs = params[i_*anc_param:i_*anc_param+anc_param]
            
            if np.sum(anc_probs) == 0:
                return np.nan
            else:
                anc_probs = (1/np.sum(anc_probs))*anc_probs
            prob_mat.append(anc_probs)
    else: 
        print("Wrong number of parameters. Mat size:",len(matrix),",Expected params:", ((2**n_anc)-1)*n_subgroups, "or", ((2**n_anc))*n_subgroups)
        print("Parameters given:", len(params))
        

    prob_mat = np.array(prob_mat)       
    prob_mat = prob_mat.T

    pmat = list([]) # Creates the matrix of probabilities for the bits to be 1.

    for reg_index in range(2**n_reg):
        ancprobvec = list([])
        for anc_index in range(n_anc):
            ancprobvec.append(np.sum(prob_mat.T[reg_index][np.where(bitmap[anc_index] == 1)]))
        pmat.append(ancprobvec)
    pmat = np.array(pmat)


    diag_subcf = 0
    offdiag_subcf = 0
    constr = 0
    for subsystem_index1 in range(2**n_reg): # loop over subsystems
        for subsystem_index2 in range(subsystem_index1,2**n_reg):

            # Gets the submatrices along the diagonal and off diagonal
            submat = matrix[subsystem_index1*n_anc:(subsystem_index1*n_anc)+n_anc, subsystem_index2*n_anc:(subsystem_index2*n_anc)+n_anc]

            if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla

                for item in itertools.product([0,1],repeat=n_anc): # Treat it as complete encoding
                    subsystem_bitstring = np.array(item) 

                    # Essentially calculating C_subsystem = <xTAx>
                    diag_subcf = diag_subcf + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*prob_mat[bitstr_index][subsystem_index1]
                    bitstr_index = bitstr_index + 1

            else: # Create pmat to obtain probabilities of off-diagonal subsystems
                sub_pmat = np.outer(pmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                offdiag_subcf = offdiag_subcf + np.sum(np.multiply(sub_pmat,submat))
                
                
#     constr = ((len(params)))*np.sum(np.multiply(params, params-1)**2) # Constraint to make variables 0 or 1
    
    return diag_subcf+(2*offdiag_subcf) + constr

class classopt_na:
        
    def __init__(self,matrix, n_ancilla, optimizer="SLSQP"): # learning rate and max iter as init attributes
        self.matrix = matrix
        self.n_ancilla = n_ancilla
        self.optimizer = optimizer

        
        if n_ancilla <= 16:
            pass
        else:
            print("MORE THAN 16 ANCILLAS. Might take too long to calculate CF in each subsystem")
            
        # All possible subsystem solutions
        self.bitmap = gen_sequences(self.n_ancilla)        
        self.subsystemsols = [item for item in itertools.product([0,1],repeat=self.n_ancilla)]
        
    def minimize_less(self, stpt=None):
        ghty1 = time.time()
                
        ######## Initial params ########
        n_subgroups = int((len(self.matrix)/self.n_ancilla))
        anc_param = (2**self.n_ancilla-1)
        nparams = (n_subgroups*anc_param)
        cflist = list([])
        ################################

        #### Initializing so that starting points fulfils the constraints ####
        if stpt == None:
            rr1 = np.random.RandomState()
        else:
            rr1 = np.random.RandomState(stpt)
        
        init_param = rr1.uniform(size =(nparams))
        init_cf = cf_classical(init_param,self.matrix,self.n_ancilla,self.bitmap)
        self.check_constr = True # Checks if constraints are satisfied after optimization
        #######################################################################
        
        ##### COBYLA/SLSQP #####
        print("Optimizer:",self.optimizer,", N_anc:",self.n_ancilla,", N_parameters:", nparams, ", N_subgroups:", n_subgroups, ", Init cf:", round(init_cf,5))

            
        def minopt(params,matrix,n_anc,bitmap):
            cf = cf_classical(params,matrix,n_anc,bitmap)
            cflist.append(cf)
            return cf
        
        bnds = [[0,1] for x in range(nparams)] # Only for L-BFGS-B, TNC, SLSQP, Powell
        
        cons = [] # Only for COBYLA, SLSQP
        l = {'type': 'ineq',
             'fun': lambda x: x - 0} # Probability has to be > 0
        u = {'type': 'ineq',
             'fun': lambda x: 1 - x} # Probability has to be < 0
        cons.append(l)
        cons.append(u)
             
        if self.optimizer == "COBYLA":
            opt = minimize(minopt,init_param,args=(self.matrix,self.n_ancilla,self.bitmap),method=self.optimizer,constraints=cons, tol=1e-8)
        else:
            opt = minimize(minopt,init_param,args=(self.matrix,self.n_ancilla,self.bitmap),method=self.optimizer,bounds=bnds, tol=1e-8)
       
        
        print(opt.message)
        finalvec = copy.deepcopy(opt.x)
        

        ###### Obtaining the final bitstring from classical measurements
        bitstr = list([]) # Final bitstring reconstructed
        for i in range(n_subgroups):
            
            # Used for getting subsystem probabilities
            subsysparam = finalvec[i*anc_param:i*anc_param+anc_param]
            subsysprob = np.abs(np.round(get_1_more(subsysparam),6))
            chosen_sol = self.subsystemsols[np.random.choice(2**self.n_ancilla,p=subsysprob)] 
            bitstr.append(chosen_sol) # Reconstruct final bitstring from subsystem solutions
    
        bitstr  = np.array(bitstr).flatten() # final bitstring
        cf_qubo = (bitstr.dot(self.matrix)).dot(bitstr)
        
        ghty2 = time.time()
        # print("Final CF from opt:", round(opt.fun,5), ", From bitstring:", round(cf_qubo,5))
    #     print("Final CF from probvec:", round(cf_classical(finalvec,self.matrix,self.n_ancilla,bitmap),5))
        print("Final bitstring:", list(map(int, bitstr)))
        print(opt.message, ", Optimization iterations:", opt.nfev, ", Time taken:", round(ghty2-ghty1,4))
        
        self.time = round(ghty2-ghty1,4)
        self.optev = opt.nfev
        self.optvec = opt.x
        self.bitstr_sol = bitstr
        self.qubo_cf = cf_qubo
        self.mincf = opt.fun
        self.cflist = cflist
                
        return opt.x
    
    
    def minimize(self, stpt=None): # Best optimizer: SLSQP
        ghty1 = time.time()
                
        ######## Initial params ########
        n_subgroups = int((len(self.matrix)/self.n_ancilla))
        anc_param = (2**self.n_ancilla)
        nparams = (n_subgroups*anc_param)
        cflist = list([])
        ################################

        #### Initializing so that starting points fulfils the constraints ####
        if stpt == None:
            rr1 = np.random.RandomState()
        else:
            rr1 = np.random.RandomState(stpt)
        
        init_param = rr1.uniform(size =nparams)
        init_cf = cf_classical(init_param,self.matrix,self.n_ancilla,self.bitmap)
        self.check_constr = True # Checks if constraints are satisfied after optimization
        #######################################################################
        
        ##### COBYLA/SLSQP #####
        print("Optimizer:",self.optimizer,", N_anc:",self.n_ancilla,", N_parameters:", nparams, ", N_subgroups:", n_subgroups, ", Init cf:", round(init_cf,5))
    
        self.nancheck = True
        
        def minopt(params,matrix,n_anc,bitmap):
            cf = cf_classical(params,matrix,n_anc,bitmap)
            cflist.append(cf)
            
            if not np.isnan(cf) and self.nancheck == True:
                return cf
            elif np.isnan(cf) and self.nancheck == True:
                self.nancheck = False
                return cf
            else:
                print("Checking if NaN has caused a problem.... Curr iter:", len(cflist))
                for i_ in range(n_subgroups):        
                    anc_probs = params[i_*anc_param:i_*anc_param+anc_param]
                
                    if np.sum(anc_probs) == 0:
                        print("Optimizer did not fix itself. Throwing error.")
                        return None
                    elif np.isnan(cflist[-1]) and np.isnan(cflist[-2]) and np.isnan(cflist[-3]):
                        print("Optimizer did not fix itself. Throwing error.")
                        return None
                    else:
                        continue
                print("Optimizer has corrected itself. Continuing with optimization")
                self.nancheck = True
                return cf

        
        bnds = [[0,1] for x in range(nparams)] # Only for L-BFGS-B, TNC, SLSQP, Powell
        
             
        if self.optimizer == "COBYLA":
            cons = [] # Only for COBYLA, SLSQP
            l = {'type': 'ineq',
                 'fun': lambda x: x - 0} # Probability has to be > 0
            u = {'type': 'ineq',
                 'fun': lambda x: 1 - x} # Probability has to be < 0
            cons.append(l)
            cons.append(u)
            
            opt = minimize(minopt,init_param,args=(self.matrix,self.n_ancilla,self.bitmap),method=self.optimizer,constraints=cons, tol=1e-8)
            
            finalvec = copy.deepcopy(opt.x)
            ###### Check 1 and 0 constraints
            finalvec = np.abs(np.round(finalvec,6))
            for prob in finalvec:
                if prob >= 0 and prob  <=1:
                    continue
                else:
                    self.check_constr = False
                    print("Probability constraint not satisifed!!", prob)
                    break
       
        else: 
            opt = minimize(minopt,init_param,args=(self.matrix,self.n_ancilla,self.bitmap),method=self.optimizer,bounds=bnds, tol=1e-8)
            finalvec = copy.deepcopy(opt.x)
        

        ###### Obtaining the final bitstring from classical measurements      
        
        bitstr = list([]) # Final bitstring reconstructed
        for i in range(n_subgroups):
            
            # Used for getting subsystem probabilities
            subsysprob = (1/np.sum(finalvec[i*anc_param:i*anc_param+anc_param]))*finalvec[i*anc_param:i*anc_param+anc_param]            
            chosen_sol = self.subsystemsols[np.random.choice(2**self.n_ancilla,p=subsysprob)]                 
            bitstr.append(chosen_sol) # Reconstruct final bitstring from subsystem solutions
    
        bitstr  = np.array(bitstr).flatten() # final bitstring
        cf_qubo = (bitstr.dot(self.matrix)).dot(bitstr)
        
        ghty2 = time.time()
        # print("Final bitstring:", list(map(int, bitstr)))
        print(opt.message, ", Optimization iterations:", opt.nfev, ", Time taken:", round(ghty2-ghty1,4))
        
        cflist = np.array(cflist)
        
        self.time = round(ghty2-ghty1,4)
        self.optev = opt.nfev
        self.optvec = opt.x
        self.bitstr_sol = bitstr
        self.qubo_cf = cf_qubo
        self.mincf = opt.fun
        self.cflist = list(cflist[~np.isnan(cflist)])
                
        return opt.x
    
    
############################################################################
############################### Matrix stuff ###############################
############################################################################

def fct(n): # Factorial function
    return np.math.factorial(n)

def binomc(n,k): # nCr function 
    return fct(n)/(fct(k)*fct(n - k))

def randsymmat5(nv, seed): # 5rd version. 
    r1 = np.random.RandomState(seed)
    matrix = np.round(r1.uniform(-1,1,size=(nv,nv)),5)
    for i in range(nv): # Off diagonal elements
        for j in range(nv):
            matrix[i][j] = matrix[j][i]
               
    r1 = np.random.RandomState(2*seed)
    for s in range(nv): # Diagonal elements
        matrix[s][s] = np.round(r1.uniform(-1,1),5)
    return matrix, str(seed)

def randgaussmat(nv, seed): # 5rd version. 
    r1 = np.random.RandomState(seed)
    matrix = np.round(r1.normal(0,1,size=(nv,nv)),5)
    for i in range(nv): # Off diagonal elements
        for j in range(nv):
            matrix[i][j] = matrix[j][i]
               
    r1 = np.random.RandomState(2*seed)
    for s in range(nv): # Diagonal elements
        matrix[s][s] = np.round(r1.normal(0,1),5)
    return matrix, "g"+str(seed)

def randsparsemat(nv,seed,sp): 
    
    # Picking the non-zero off diagonal entries
    rs1 = np.random.RandomState(seed)
    matrix = np.zeros((nv,nv))
    upp_diag = int((nv**2)/2 - (nv/2)) # number of elements in upper diagonal. Same as above
    ind_list = rs1.choice(upp_diag,np.int(sp*upp_diag), replace=False)
    
    randindex = list([])
    
    k = 0
    for item in itertools.combinations([x for x in range(nv)], r=2):       
        if k in ind_list:
            randindex.append(list(item))
        else:
            pass
        k = k +1
    
    # Setting the values the non-zero off diagonal entries
    for item in randindex:
        drawrand = np.round(rs1.uniform(-1,1),5)
        matrix[item[0],item[1]] = drawrand
        matrix[item[1],item[0]] = drawrand
        
        
    # Picking the non-zero diagonal entries
    rs2 = np.random.RandomState(2*seed)
    ind_list = rs2.choice(nv,np.int(sp*nv), replace=False)
    
    # Setting the non-zero diagonal entries
    k=0
    for s in range(nv):
        if k in ind_list:
            matrix[s][s] = np.round(rs2.uniform(-1,1),5)
        else:
            pass
        k = k+1
   
    return matrix, "sp"+str(seed) + "."+str(int(sp*10))

def rand_sp_int_mat(nv,seed,sp): 
    
    #### sp is the percentage of non-zero elements. Bigger number: more dense. 1: Fully dense, ~0: very sparse
    spp = str(int(sp*10))
    
    #### Picking the non-zero off diagonal entries
    rs1 = np.random.RandomState(seed)
    matrix = np.zeros((nv,nv))
    upp_diag = int((nv**2)/2 - (nv/2)) # number of elements in upper diagonal. Same as above
    ind_list = rs1.choice(upp_diag,int(sp*upp_diag), replace=False)
    
    randindex = list([])
    
    k = 0
    for item in itertools.combinations([x for x in range(nv)], r=2):       
        if k in ind_list:
            randindex.append(list(item))
        else:
            pass
        k = k +1
    
    # Setting the values the non-zero off diagonal entries
    for item in randindex:
        drawrand = rs1.randint(-5,5)
        while drawrand == 0: # While loop to prevent drawing extra zeros
            drawrand = rs1.randint(-5,5)
        matrix[item[0],item[1]] = drawrand
        matrix[item[1],item[0]] = drawrand
        
        
    # Picking the non-zero diagonal entries
    rs2 = np.random.RandomState(2*seed)
    ind_list = rs2.choice(nv,int(sp*nv), replace=False)
    
    # Setting the non-zero diagonal entries
    k=0
    for s in range(nv):
        if k in ind_list:
            drawrand = rs2.randint(-5,5)
            while drawrand == 0: # While loop to prevent drawing extra zeros
                drawrand = rs2.randint(-5,5)
            matrix[s][s] = drawrand
            
        else:
            pass
        k = k+1
   
    return matrix, "sii"+str(seed) + "." + str(spp)


def get_mat_sp(matrix): # Gets the "sp" value from the matrix.
    unique, counts = np.unique(matrix, return_counts=True)
    tt1 = dict(zip(unique, counts))
    try:
        mm1 = 1-(tt1[0]/np.sum(counts)) # ratio of zeros to total number of elements
    except KeyError:
        mm1 = 0
        
    return np.round(mm1,5)


def quickmax_mat(matrix,anc, matrixname): # Sorting the matrix depends on the number of ancillas    
    matrix = np.array(matrix)
    
    
    try:
        if matrixname.startswith("sp") or matrixname.startswith("pm"):
            matname = "sm"+matrixname[2:]+"a"+str(int(anc))
        else:
            matname = "sm"+matrixname+"a"+str(int(anc))
    except:
        matname = matrixname
    
    def remove_from_mat(mat1,rowcol): # Removes all element in the chosen row and the same column
        mat = copy.deepcopy(mat1)
        mat[rowcol,:] = 0
        mat[:,rowcol] = 0
        return mat

    n_subsystem = int(len(matrix)/anc)
    
    hmat = np.abs(copy.deepcopy(matrix))
    hmat[range(len(hmat)),range(len(hmat))] = 0 # remove diagonal elements.
    
    diag_element_sort = list([])
    for sub_idx in range(n_subsystem):
        remove_elements = np.argwhere(np.triu(hmat,k=1) == np.max(np.triu(hmat,k=1)))[0] # Finds the biggest element in off-diag
         # Saves the location so that it can be first element in subsystem
            
# #         print("With elite sort")
        omat1 = copy.deepcopy(hmat)
        omat2 = copy.deepcopy(hmat)
        re1 =  copy.deepcopy(remove_elements)
        re2 = copy.deepcopy(remove_elements)[::-1]

        nmat = remove_from_mat(omat1,remove_elements[0])
        trymat = remove_from_mat(omat2,re2[0])
        
        g1sort = list([])
        g2sort = list([])
        for na_idx in range(anc-2): # Finds the next biggest element to put in subsystem. Loops until subsystem is filled
#             nmatsum = nmatsum + np.max(nmat[re1[1]])
            re1 =np.concatenate(([re1[1]],np.argwhere(nmat[re1[1]] == np.max(nmat[re1[1]]) )[0]))
            g1sort.append(re1)
            nmat = remove_from_mat(nmat,re1[0])
            
#             tmatsum = tmatsum + np.max(trymat[re2[1]])
            re2 =np.concatenate(([re2[1]],np.argwhere(trymat[re2[1]] == np.max(trymat[re2[1]]) )[0]))
            g2sort.append(re2)
            trymat = remove_from_mat(trymat,re2[0])
        
        g11sort = np.array(g1sort).flatten()
        _, g1_idx  = np.unique(g11sort, return_index=True) # Sorts the matrix according to the order of the elements removed
        g11_ = g11sort[np.sort(g1_idx)]
        
        g22sort = np.array(g2sort).flatten()
        _, g2_idx  = np.unique(g22sort, return_index=True) # Sorts the matrix according to the order of the elements removed
        g22_ = g22sort[np.sort(g2_idx)]
        
        nmatsum = np.sum(np.abs(matrix[np.ix_(list(g11_),list(g11_))]))
        tmatsum = np.sum(np.abs(matrix[np.ix_(list(g22_),list(g22_))]))
        
        if tmatsum > nmatsum:
            diag_element_sort.append(remove_elements[::-1])
            trymat = remove_from_mat(trymat,re2[1])
            hmat = copy.deepcopy(trymat)
            for item in g2sort:
                diag_element_sort.append(item)
        else:
            diag_element_sort.append(remove_elements)
            nmat = remove_from_mat(nmat,re1[1])
            hmat = copy.deepcopy(nmat)
            for item in g1sort:
                diag_element_sort.append(item)
                
    diag_element_sort = np.array(diag_element_sort).flatten()
    
    _, idx  = np.unique(diag_element_sort, return_index=True) # Sorts the matrix according to the order of the elements removed
    diag_element_sort = diag_element_sort[np.sort(idx)]
    diag_element_sort = np.concatenate((diag_element_sort, np.setdiff1d(np.arange( len(matrix) ),diag_element_sort) ))
#     print(diag_element_sort)
    
    return matrix[diag_element_sort,:][:,diag_element_sort], matname # Returns new matrix


def brutemax_mat(matrix,na): 
    # Always does better than other sort
    # If bruteforce does not do better, remember to remove diag elements during comparison
    matrix = np.array(matrix)
    hmat = copy.deepcopy(np.abs(matrix)) # consider magnitude of elements only
    hmat[range(len(matrix)),range(len(matrix))] = 0 # Set diagonals to 0
    
    diag_element_sort = list([])
    sublist = list([])
    
    elements_to_sort = np.arange(len(matrix))
    
    mmjk = 0
    while True:
        init_sum = -1e9
        mmjk = mmjk +1 
        for item in itertools.combinations(elements_to_sort,r=na):
            subsys_sum = np.sum(hmat[np.ix_(list(item),list(item))])

            if subsys_sum > init_sum:
                init_sum = subsys_sum
                group_of_var = list(item)

        sublist.append(group_of_var)
        elements_to_sort = np.setdiff1d(elements_to_sort,group_of_var)

        if len(elements_to_sort) < na:
            break
    
    diag_element_sort = np.array(sublist).flatten()
    _, idx  = np.unique(diag_element_sort, return_index=True) # Sorts the matrix according to the order of the elements removed
    diag_element_sort = diag_element_sort[np.sort(idx)]
    diag_element_sort = np.concatenate((diag_element_sort, np.setdiff1d(np.arange( len(hmat) ),diag_element_sort) ))

    return matrix[diag_element_sort,:][:,diag_element_sort]


def blockmat_nosort(nc, seed, block_size):
    
    matrix = np.zeros(shape=(nc,nc))
    nblocks = nc//block_size          # Number of blocks
    rblock_size = nc%block_size       # Size of last block
    
    bz1 = np.random.RandomState(seed)
    for nb in range(nblocks):                              # Fill up the matrix in blocks
        for i in range(block_size):
            for j in range(i, block_size):
                if i == j:
                    continue
                randval = bz1.uniform(-1,1)
                matrix[nb*block_size+i, nb*block_size+j] = randval
                matrix[nb*block_size+j, nb*block_size+i] = randval
    
    for i in range(rblock_size):                          # Fill the last block of the matrix
        for j in range(i, rblock_size):
            if i==j:
                continue
            randval = bz1.uniform(-1,1)
            matrix[nblocks*block_size+i, nblocks*block_size+j] = randval
            matrix[nblocks*block_size+j, nblocks*block_size+i] = randval
            
    bz2 = np.random.RandomState(seed)
    for i in range(nc):                                   # Diagonal elements in matrix
        randval = bz2.uniform(-1,1)
        matrix[i,i] = randval
    
    return matrix, "bn"+str(seed) + "." + str(block_size) # "n" in name means no-sorting will be done 

def blockmat_sort(nc, seed, block_size):
    
    matrix = np.zeros(shape=(nc,nc))
    nblocks = nc//block_size          # Number of blocks
    rblock_size = nc%block_size       # Size of last block
    
    bz1 = np.random.RandomState(seed)
    for nb in range(nblocks):                              # Fill up the matrix in blocks
        for i in range(block_size):
            for j in range(i, block_size):
                if i == j:
                    continue
                randval = bz1.uniform(-1,1)
                matrix[nb*block_size+i, nb*block_size+j] = randval
                matrix[nb*block_size+j, nb*block_size+i] = randval
    
    for i in range(rblock_size):                          # Fill the last block of the matrix
        for j in range(i, rblock_size):
            if i==j:
                continue
            randval = bz1.uniform(-1,1)
            matrix[nblocks*block_size+i, nblocks*block_size+j] = randval
            matrix[nblocks*block_size+j, nblocks*block_size+i] = randval
            
    bz2 = np.random.RandomState(seed)
    for i in range(nc):                                   # Diagonal elements in matrix
        randval = bz2.uniform(-1,1)
        matrix[i,i] = randval
    
    return matrix, "b"+str(seed) + "." + str(block_size) # "n" in name means no-sorting will be done 


    
    
    
    