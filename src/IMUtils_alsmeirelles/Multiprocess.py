#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os
import numpy as np
import multiprocessing
import concurrent.futures

from tqdm import tqdm

def multiprocess_run(exec_function,exec_params,data,cpu_count,pbar,step_size,output_dim=1,split_output=False,txt_label='',verbose=False):
    """
    Runs exec_function in a process pool. This function should receive parameters as follows:
    (iterable_data,param2,param3,...), where paramN is inside exec_params

    This multiprocessing solution process the full data as a hole, not regarding eventual inner subdivisions.
    If output should be splited based on inputs, it will be a dictionary and data should be hashable

    @param exec_function <function>
    @param exec_params <tuple> or None
    @param data <iterable>
    @param cpu_count <int>: use this number of cores
    @param pbar <boolean>: user progress bars
    @param step_size <int>: size of the iterable that exec_function will receive
    @param output_dim <int>: exec_function produces how many sets of results?
    """

    # Perform extractions of frames in parallel and in steps
    step = int(len(data) / step_size) + (len(data)%step_size>0)
    if split_output:
        datapoints_db = {}
    else:
        datapoints_db = [[] for i in range(output_dim)]
    semaphores = []

    process_counter = 0
    pool = multiprocessing.Pool(processes=cpu_count,maxtasksperchild=50,
                                    initializer=tqdm.set_lock, initargs=(multiprocessing.RLock(),))

    if pbar:
        l = tqdm(desc="Processing {0}...".format(txt_label),total=step,position=0)

    datapoints = np.asarray(data)
    if exec_params is None:
        exec_params = ()

    for i in range(step):
        # get a subset of datapoints
        end_idx = step_size

        if end_idx > len(data):
            end_idx = len(data)

        cur_datapoints = datapoints[:end_idx]

        if pbar:
            semaphores.append(pool.apply_async(exec_function,
                                args=(cur_datapoints,) + exec_params,
                                callback=lambda x: l.update(1)))
        else:
            semaphores.append(pool.apply_async(exec_function,
                                args=(cur_datapoints,) + exec_params))

        datapoints = np.delete(datapoints,np.s_[:end_idx],axis=0)

        if pbar:
            if process_counter == cpu_count+1:
                semaphores[process_counter].wait()
                process_counter = 0
            else:
                process_counter += 1

        #datapoints = np.delete(datapoints,np.s_[i*step_size : end_idx],axis=0)
        #del cur_datapoints

    for i in range(len(semaphores)):
        res = semaphores[i].get()
        if split_output:
            datapoints_db[i] = res
        else:
            for k in range(output_dim):
                datapoints_db[k].extend(res[k])
        if not pbar and verbose > 0:
            print("[{2}] Done transformations (step {0}/{1})".format(i,len(semaphores)-1,txt_label))
            sys.stdout.flush()

    if pbar:
        l.close()
        print("\n"*cpu_count)

    #Free all possible memory
    pool.close()
    pool.join()

    del datapoints

    # remove None points
    if split_output:
        return datapoints_db
    else:
        return tuple(filter(lambda x: not x is None, datapoints_db))
    
