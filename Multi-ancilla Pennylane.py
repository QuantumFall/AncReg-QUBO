# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:46:39 2021

@author: Benjamin
"""

import os
file_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_loc)

import pennylane as qml
from pennylane import numpy as plnp
import numpy as np
import time
import re
import copy

import multiprocessing as mp
import itertools


    
# def gen_sequences(N): 
#     # Slightly quicker way of producing the bitmap_t matrix, but still not usable for n_a = 32
#     # Replaces bitmap_t = np.array([np.array(bitstr) for bitstr in itertools.product([0,1],repeat=n_ancilla)]).T
#     sequences = []
#     for m in reversed(range(N)):
#         power = 2**m
#         sequence = ([0] * power + [1] * power) * (2**N//(2*2**m))
#         sequences.append(sequence)
#     return sequences

def gen_sequences(N):
    # Superfast method of producing the bitmap_t matrix. Much faster than previous methods
    cube = np.zeros([2]*N)
    cube[1, ...] = 1

    return np.array([
        np.moveaxis(cube, 0, i).ravel()
        for i in range(N)
    ])

def HEcirc(params, wires = None, nq = None, depth = None, sample=False):

    # Apply one layer of hadamards
    for nq_ in range(nq):
        qml.Hadamard(wires=nq_)
        
    # Apply layers of single qubit rotations and entangling gates
    for l_ in range(depth):
        for i_ in range(nq):
            qml.RY(params[i_ + l_*nq], wires=i_)

        for j in range(0,nq,2):
            if j >= nq-1:
                continue
            else:
                qml.CNOT(wires=[j,j+1])
        for k in range(1,nq,2):
            if k >= nq-1:
                continue
            else:
                qml.CNOT(wires=[k,k+1])
        
    if sample is True: # measurement phase
        return tuple([qml.sample(qml.PauliZ(i)) for i in range(nq)])
    else:
        return qml.probs(wires=[i for i in range(nq)])
    
##########################################################################

def sciopt(single_input):

    matfilename, nanc, dp, stpt, n_measurements, stepsize  = single_input
    bitmap = gen_sequences(nanc)
    
    # Load matrices
    matseed = re.findall('^nc(\d+)-mat(.*).npy$', matfilename)[0][1]
    loadmatd = np.load(matfilename, allow_pickle = True).item()
    
    amat_load = loadmatd['matrix']
    matmin = loadmatd['matmin']
    # matmax = loadmatd['matmax']
    
    amat, _ = quickmax_mat(amat_load,nanc,matseed)
    nq = int(np.ceil(np.log2(len(amat)/nanc)) + nanc)

    dev = qml.device("lightning.qubit", wires=nq, shots=n_measurements)                # Initialize pennylane device
    MyQnode = qml.QNode(HEcirc,dev,diff_method='best')
    

    ######################################################################

    def getproj(probs,na,nr):
    # Works for both finite and infinite measurements
    
        nstates_anc = 2**na
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
        return plnp.array(regprob), plnp.array(proj)
    

    
    def anc_cf(params, matrix, nqubits, n_ancilla, depth, bitmap):
        # Works for both finite and infinite measurements
        # Unmeasured registers will give equal probability to all ancilla states
    
        probs = MyQnode(params, nq = nqubits,  depth = depth, sample=False)
        n_reg = nqubits - n_ancilla # Obtains number of register qubits
    
        reg_prob, reg_anc_prob = getproj(probs,n_ancilla,n_reg) # Get register and ancilla probabilities from input state
        reg_anc_prob = copy.deepcopy(reg_anc_prob.T)
        
        anc_prob = []
        for reg_idx in range(2**n_reg):
            if reg_prob[reg_idx] == 0:
                anc_prob.append(np.full(2**n_ancilla,1/(2**n_ancilla)))
            else:
                anc_prob.append(reg_anc_prob[reg_idx]/reg_prob[reg_idx])
        anc_prob = plnp.array(anc_prob)

        # Transposed List of all possible bitstrings from ancilla qubits. Used to construct pmat. 
        #===============================================================
        #============= WiLL be long if n_a is large (>16) ==============
        #===============================================================
    
        
        # Pmat is a rectangular matrix. For 2anc, each row is [c^2 + d^2, b^2 + d^2]
        # The same pmat is used for all subsystems so it only has to be generated once
        pmat = list([]) # Creates the matrix of probabilities for the bits to be 1.
        
        for reg_index in range(2**n_reg):
            ancprobvec = list([])
            for anc_index in range(n_ancilla):
                ancprobvec.append(plnp.sum(anc_prob[reg_index][np.where(bitmap[anc_index] == 1)]))
            pmat.append(ancprobvec)
        pmat = plnp.array(pmat)
        
        diag_subcf = 0
        offdiag_subcf = 0
    
        for subsystem_index1 in range(2**n_reg): # loop over subsystems
            for subsystem_index2 in range(subsystem_index1,2**n_reg):
    
                # Gets the submatrices along the diagonal and off diagonal
                submat = matrix[subsystem_index1*n_ancilla:subsystem_index1*(n_ancilla)+n_ancilla, subsystem_index2*n_ancilla:subsystem_index2*(n_ancilla)+n_ancilla]
    
                if subsystem_index1 == subsystem_index2: # For subsystems along the diagonal
                    bitstr_index = 0 # Count through all possible ancilla bitstrings to get the correct probability index in the ancilla
                    
                    for item in itertools.product([0,1],repeat=n_ancilla): # Treat it as complete encoding
                        subsystem_bitstring = plnp.array(item) 
                        
                        # Essentially calculating C_subsystem = <xTAx>
                        diag_subcf = diag_subcf + (subsystem_bitstring.dot(submat)).dot(subsystem_bitstring)*anc_prob.T[bitstr_index][subsystem_index1]
                        bitstr_index = bitstr_index + 1
    
                else: # Create pmat to obtain probabilities of off-diagonal subsystems
                    sub_pmat = np.outer(pmat[subsystem_index1],pmat[subsystem_index2]) # Gets the correct probabilities for the subsystems
                    offdiag_subcf = offdiag_subcf + np.sum(np.multiply(sub_pmat,submat)) # calculates cost function using hadamard product
    
        offdiag_subcf = offdiag_subcf*2 # Multiply by 2 since calculations were done for only upper diagonal matrix
        return diag_subcf+offdiag_subcf # Sum diagonal and off diagonal terms

    ######################################################################
    
    # Initialize random parameters and optimizer
    zs1 = np.random.RandomState(seedlist[stpt])
    initvec = plnp.array(zs1.uniform(0,2*np.pi, dp * nq))
    init_cf = np.float64(anc_cf(initvec, matrix = amat, nqubits = nq, n_ancilla=nanc, depth = dp, bitmap = bitmap))
    theta = copy.deepcopy(initvec)


    # Optimizer has to be initialized as close to the inner loop as possible as it keeps track of past iterations.
    # Otherwise, separate optimization runs (from different starting points) might be compromised by previous runs.
    opt = qml.AdamOptimizer(stepsize=stepsize)
    n_iter = 200     

    cflist = list([])
    thetalist = list([])
    cflist.append(init_cf)
    thetalist.append(initvec)
    

    print(f"Begin opt. Stpt: {stpt:<3d}, nc: {len(amat):<2d}, na: {nanc:<2d}, nq: {nq:<2d}, nd: {dp:<2d}",
          "shots: ", n_measurements,
          f", init CF: {init_cf:<3.2f}, matmin: {matmin:<3.2f},")


    tlt1 = time.time()

    for t_ in range(1,n_iter+1):
    
        theta, intermediate_cf  = opt.step_and_cost(anc_cf, theta, matrix=amat, nqubits = nq, n_ancilla = nanc, depth = dp, bitmap = bitmap)
        cflist.append(np.float64(intermediate_cf))
        thetalist.append(np.array(theta))
    
        if t_%40 == 0:                                                          # Print output every 20 iterations
            time_elapased = time.time() - tlt1
            time_per_iter = time_elapased/t_
            time_remaining = time_per_iter * (n_iter-t_)
            total_time = time_remaining + time_elapased

            print(f"Stpt: {stpt:<2d}, nc: {len(amat):<2d}, na: {nanc:<1d}, nq: {nq:<1d}, nd: {dp:<1d}",
                  "shots: ", n_measurements,
                  f", init CF: {init_cf:<3.2f}, matmin: {matmin:<3.2f},",
                  f"curr CF: {intermediate_cf:<3.2f},",
                  f"curr iter: {t_:<3d}, t: {time_elapased:<2.1f}, t_to_end: {time_remaining:<2.1f}, t_tot: {total_time:<2.1f}") 

    time_taken = time.time()- tlt1
    
    cflist.append(min(cflist))
    thetalist.append(thetalist[np.argmin(cflist)])

    print(f"Completed Stpt: {stpt:<3d}, nc: {len(amat):<2d}, na: {nanc:<2d}, nq: {nq:<2d}, nd: {dp:<2d}",
          "shots: ", n_measurements,
          f", init CF: {init_cf:<3.2f}, matmin: {matmin:<3.2f},",
          f"final CF: {cflist[-1]:<3.2f}, t: {time_taken:<2.1f}")

    file1 = "CFqml-nc"+str(len(amat))+"-"+str(matseed)+" - na" +str(nanc)+"lr"+str(stepsize)+".npy"
    file2 = "Optvqml-nc"+str(len(amat))+"-"+str(matseed)+" - na" +str(nanc)+"lr"+str(stepsize)+".npy"

    # Blank file used to indicate that process is done and queueing to modify data file
    curr_process_file = "callback_savefile"+str(stpt)+"-"+str(matseed)+"-"+str(dp)+str(nanc)+str(time.time()%1)[-6:]+str(zs1.randint(9999,99999))+".npy"          
    np.save(curr_process_file,np.array([]))                    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    search_k = 1
    while search_k > 0:

        fdict = dict({})
        
        for cfile in os.listdir(os.chdir(file_loc)):                     # Check if others are also queueing
            if cfile.startswith("callback_savefile"): 
                try:
                    file_time = copy.deepcopy(os.stat(cfile).st_ctime)
                    fdict[file_time] = copy.deepcopy(cfile)
                except FileNotFoundError:
                    print(f"{'':=^94}\n",
                          "File not found error encountered", curr_process_file, cfile,
                          f"\n{'':=^94}")
                    pass
                except PermissionError:
                    print(f"{'':=^94}\n",
                          "File not found error encountered", curr_process_file, cfile,
                          f"\n{'':=^94}")
                    pass
            

        comes_first = np.min(list(fdict.keys()))
        search_k += 1

        if search_k > 2 and fdict[comes_first] == curr_process_file: # Break out of queue if it's your turn
            search_k = -1
            print("\n###-----> My turn to save file", curr_process_file)
        else:
            if search_k > 20 and search_k%9 == 0:
                print("++++++++ Waiting in queue to save data ++++++++", curr_process_file, "\nCurrent first in line:",  fdict[comes_first])
            if search_k%5 == 0:
                checkk = list(fdict.values())
                if curr_process_file in checkk:
                    pass
                else:
                    print("\nXXXXX----->>> QUEUE TICKET IS MISSING!!! ", curr_process_file, "\n")
                    print(list(fdict.values()))
                    break
            # Continue waiting if not your turn. Longer the better as it takes time to write to file
        time.sleep(0.1 + 0.023 * len(fdict)) # If too short, another process might start trying to load file even though it is being written to.
            
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Modifies the saved data file
    if os.path.isfile(os.path.join(os.getcwd(), file1)):

        cf_dat = np.load(os.path.join(os.getcwd(), file2), allow_pickle = True).item()
        param_dat = np.load(os.path.join(os.getcwd(), file1), allow_pickle = True).item()

        if dp not in cf_dat:
            cf_dat[dp] = dict({})
            param_dat[dp] = dict({})
        if n_measurements not in cf_dat[dp]:
            cf_dat[dp][n_measurements] = dict({})
            param_dat[dp][n_measurements] = dict({})
            
        cf_dat[dp][n_measurements][stpt] = copy.deepcopy(cflist)
        param_dat[dp][n_measurements][stpt] = copy.deepcopy(thetalist)

        np.save(file2,cf_dat)
        np.save(file1,param_dat)
        print("Existing file found. Data saved in existing file:", file2,"\n",
        "Number of stpts saved:", len(cf_dat), curr_process_file)

    else:

        cf_dat = dict({})
        param_dat = dict({})
        
        if dp not in cf_dat:
            cf_dat[dp] = dict({})
            param_dat[dp] = dict({})
        if n_measurements not in cf_dat[dp]:
            cf_dat[dp][n_measurements] = dict({})
            param_dat[dp][n_measurements] = dict({})
            
        cf_dat[dp][n_measurements][stpt] = copy.deepcopy(cflist)
        param_dat[dp][n_measurements][stpt] = copy.deepcopy(thetalist)

        np.save(file2,cf_dat)
        np.save(file1,param_dat)
        

        print("Creating new file to save data:", file2,"\n",
        "Number of stpts saved:", len(cf_dat), curr_process_file)
        
    print("#####---> Cleaning up saved file", curr_process_file)
    os.remove(curr_process_file)   
    print("######--> Cleaned up saved file", curr_process_file, "\n")

    return cflist


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
            
        omat1 = copy.deepcopy(hmat)
        omat2 = copy.deepcopy(hmat)
        re1 =  copy.deepcopy(remove_elements)
        re2 = copy.deepcopy(remove_elements)[::-1]

        nmat = remove_from_mat(omat1,remove_elements[0])
        trymat = remove_from_mat(omat2,re2[0])
        
        g1sort = list([])
        g2sort = list([])
        for na_idx in range(anc-2): # Finds the next biggest element to put in subsystem. Loops until subsystem is filled
            re1 =np.concatenate(([re1[1]],np.argwhere(nmat[re1[1]] == np.max(nmat[re1[1]]) )[0]))
            g1sort.append(re1)
            nmat = remove_from_mat(nmat,re1[0])
            
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
    
    return matrix[diag_element_sort,:][:,diag_element_sort], matname # Returns new matrix


n_stpt = 20                                                          # Number of starting points
rseed = np.random.RandomState(432)
seedarr = np.arange(1,99999)
rseed.shuffle(seedarr)
seedlist = {x:seedarr[x] for x in range(1,n_stpt + 1)}


def main():
    
    # Purpose of code is to prevent one very long starting point from "hogging" the system
    # Any process that is done will just save the data and move on rather than wait for the long one to finish
    
    lr = 0.02
    matlistt = ["nc64-matsi6123.9.npy"]
    na_list = [1,2,4]
    depthlist = [2,4,6,8,10]
    nmeaslist = [None,100,1000,10000,100000]
    
    mp_input = list([])
    
    for mat__ in matlistt:
        nc, matseed = re.findall('^nc(\d+)-mat(.*).npy$', mat__)[0]
          
        for n_anc in na_list: # number of ancillas
            file1 = "CFqml-nc"+str(nc)+"-"+str(matseed)+" - na" +str(n_anc)+"lr"+str(lr)+".npy"
            
            try:
                load_dat = np.load(file1, allow_pickle = True).item()
            except FileNotFoundError:
                pass
            
            for cdepth in depthlist: 
                for n_measurements in nmeaslist:
                    
                    try:
                        load_dat[cdepth][n_measurements]
                        for seed in seedlist: 
                            if seed in load_dat[cdepth][n_measurements]:
                                continue
                            mp_input.append([mat__, n_anc, cdepth, seed, n_measurements, lr])

                    except (KeyError, NameError):
                        for seed in seedlist: 
                            mp_input.append([mat__, n_anc, cdepth, seed, n_measurements, lr])


    print(f"{'':=^64}\n",
          f"{' Total tasks: %.d ':^64}" %(len(mp_input)),
          f"\n{'':=^64}")                   

    gyp1 = time.time()
    pool = mp.Pool(4) # mp.cpu_count()
    pool.map(sciopt,mp_input)
    pool.close()
    pool.join()
    
       
    print(f"{'':=^64}\n",
          f"{' Time taken: %.3f ':^64}" %(time.time()-gyp1),
          f"\n{'':=^64}")

     
    
if __name__ == '__main__':
    print()
    print("Starting script....")
    print()
    
    for cfile in os.listdir(os.chdir(file_loc)):                     # Check if others are also queueing
        if cfile.startswith("callback_savefile"): 
            os.remove(cfile)  
    
    krib1 = time.time()
    main()
    print("#################################################################")
    print("Time taken to for entire script: %.3f" %(time.time()-krib1))

    
    
    
    
    

