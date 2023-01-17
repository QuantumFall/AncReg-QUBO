# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:46:39 2021

@author: Benjamin
"""

import os
file_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_loc)

import numpy as np
import time
import re

import NaSGD as na
from scipy.optimize import minimize
import multiprocessing as mp




def anglevector(seed,n_q, n_layer):
    
    if seed == 0:
        return np.zeros(n_q*n_layer)
    elif seed == "random":
        return np.random.uniform(0,2* np.pi,n_q*n_layer)
    else:
        zz1 = np.random.RandomState(seed)
        return zz1.uniform(0,2* np.pi,n_q*n_layer)
    


def sciopt(single_input):
    
    
    matfilename, nanc, dp, bitmap, seed, copt = single_input
      
    # Load matrices
    matseed = re.findall('^nc(\d+)-mat(.*).npy$', matfilename)[0][1]
    loadmatd = np.load(matfilename, allow_pickle = True).item()
    
    amat_load = loadmatd['matrix']
    matmin = loadmatd['matmin']
    # matmax = loadmatd['matmax']
    
    amat, _ = na.quickmax_mat(amat_load,nanc,matseed)
    nq = int(np.ceil(np.log2(len(amat)/nanc)) + nanc)
    
    stpt = seedlist[seed]
    
    cflist = list([])
    optvec = list([])

    
    def vqee(angles, matrix, nnq, nnanc, circuit_depth, bitmap): # Function to be minimized
        
        hardware_efficient_circuit = na.circuit(nnq, circuit_depth, angles) 
        circuit_state = na.statevec(hardware_efficient_circuit)
        cfval = na.anc_cf_inf(circuit_state, matrix, nnanc,bitmap)
        
        # For saving data
        cflist.append(cfval)
        optvec.append(angles)
        
        curr_iter = len(cflist)
        if curr_iter%200 == 0:
            print("Startpoint: %.d. Current iter: %.d, CF val: %.3f" %(seed, curr_iter, cfval ))
        
        return cfval
    
    # For  saving data
    initvec = anglevector(stpt,nq, dp)
    initcirc = na.circuit(nq, dp, initvec) 
    initcf = na.anc_cf_inf(na.statevec(initcirc), amat, nanc,bitmap)
    cflist.append(initcf)
    optvec.append(initvec)
    
    time.sleep(np.random.uniform(0.1,2))

    print(f"Begin Opt. Stpt: {seed:<2d}, nc: {len(amat):<2d}, na: {nanc:<1d}, nq: {nq:<1d}, nd: {dp:<1d},",
          "shots: inf",
          f", init CF: {initcf:<3.2f}, matmin: {matmin:<3.2f}")

    options_ = {'maxiter': 1000}
    cray1 = time.time()
    opt = minimize(vqee, initvec,args=(amat, nq, nanc, dp, bitmap),method=copt, options=options_)
    
    cflist.append(opt.fun)
    optvec.append(opt.x)
    time_taken = time.time()- cray1
    
    file1 = "CF-"+copt+"v1-nc"+str(len(amat))+"-"+str(matseed)+" - na" +str(nanc)+ "dp"+str(dp)+"minf.npy"                            
    file2 = "Optv-"+copt+"v1-nc"+str(len(amat))+"-"+str(matseed)+" - na" +str(nanc)+ "dp"+str(dp)+"minf.npy"
    

    print(f"Completed. Stpt: {seed:<2d}, nc: {len(amat):<2d}, na: {nanc:<1d}, nq: {nq:<1d}, nd: {dp:<1d},",
          "shots: inf",
          f", init CF: {initcf:<3.2f}, matmin: {matmin:<3.2f},",
          f"final CF: {cflist[-1]:<3.2f}, t: {time_taken:<2.1f}s, cf_calls: {len(cflist):<5d}")
    
    
    # Blank file used to indicate that process is done and queueing to modify data file
    curr_process_file = "callback_savefile"+str(seed)+".npy"          
    np.save(curr_process_file,np.array([]))                    
    
    # Queue system needed to prevent two processes that finish close together from overwriting each other's data
    search_k = 1
    while search_k > 0:
        fdict = dict({})
        
        for filename in os.listdir(os.getcwd()):                     # Chech if others are also queueing
            if filename.startswith("callback_savefile"): 
                fdict[np.round(os.stat(filename).st_ctime,9)] = filename
                continue
            else:
                continue
            
        comes_first = np.min(list(fdict.keys()))                     # Chech epoch time to see who came first
        search_k += 1
        if search_k > 2 and fdict[comes_first] == curr_process_file: # Break out of queue if it's your turn
            search_k = -1
        else:                                                        # Continue waiting if not your turn
            time.sleep(1.2)
    
    # Modifies the saved data file
    if os.path.isfile(file1):
        cf_d = np.load(file1, allow_pickle = True).item()
        optv_d = np.load(file2, allow_pickle = True).item()
        
        cf_d[seed] = cflist
        optv_d[seed] = optvec
    
        np.save(file1, cf_d)
        np.save(file2, optv_d)
        print("Existing file found. Data saved in existing file:", file1,
        "Number of stpts saved:", len(cf_d),"\n")

    else:
        cf_d = dict({})
        optv_d = dict({})

        cf_d[seed] = cflist
        optv_d[seed] = optvec
        
        np.save(file1, cf_d)
        np.save(file2, optv_d)
        print("Creating new file to save data:", file1,
        "Number of stpts saved:", len(cf_d),"\n")
        
    os.remove("callback_savefile"+str(seed)+".npy")                 # Remove place in queue so others can go

    return True




seedlist =  dict({1:1789, 2:145, 3:2736, 4:2223, 5:3109, 6:3810, 7:4129 , 8:494 ,9:5656, 
                    10:541 ,11:6861 ,12: 660, 13:78, 14:7209, 15:8998, 16:8214, 17:990, 18:92, 19:398, 20:1143})



def main():
    
    # Purpose of code is to prevent one very long optimization run point from "hogging" the system
    # Any process that is done will just save the data and move on rather than wait for the rest to finish
    
    c_opt = "COBYLA"
    
    matlistt = ["nc64-matsi6123.9.npy"]
    
    mp_input = list([])
    
    for mat__ in matlistt:
        for n_anc in [1, 2, 4]: # number of ancillas
            bmap = na.gen_sequences(n_anc)
            for cdepth in [2,4]: 
                
                
                for seed in seedlist: 
                    mp_input.append([mat__, n_anc, cdepth, bmap, seed, c_opt])
                    
    print(f"{'':=^64}\n",
          f"{' Total tasks: %.d ':^64}" %(len(mp_input)),
          f"\n{'':=^64}") 
                
    pool = mp.Pool(4) # mp.cpu_count()
    pool.map_async(sciopt,mp_input)
    pool.close()
    pool.join()

        
    
if __name__ == '__main__':
    print()
    print("Starting script.... Quantum circuit, infinite measurements, scipy optimizer")
    print()
    na.loadstat()
    
    for cfile in os.listdir(os.chdir(file_loc)):                     # Check if others are also queueing
        if cfile.startswith("callback_savefile"): 
            os.remove(cfile)  
    
    krib1 = time.time()
    main()
    print("#################################################################")
    print("Time taken to for entire script: %.3f" %(time.time()-krib1))

    
    
    
    
    

