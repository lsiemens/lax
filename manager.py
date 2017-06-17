#! /usr/bin/env python3

import lax

import os
import time
import multiprocessing

class manager:
    def __init__(self, fname_base, processes=1, status_check=120):
        self.fname_base = fname_base
        self.processes = processes
        self.status_check = status_check

        self.jobs = None
        
    def start(self):
        jobid = 0
        self.jobs = []
        init_seed = time.time()

        while True:
            terminated_ids = []
            for process, id in self.jobs:
                if os.path.isfile(self.fname_base + str(id) + ".stktrc"):
                    terminated_ids.append(id)
        
            self.jobs = [(process, id) for process, id in self.jobs if id not in terminated_ids]
        
            while len(self.jobs) < self.processes:
                print("starting process: " + str(jobid))
                fname = self.fname_base + str(jobid) + ".dat"
                dname = self.fname_base + str(jobid) + ".stktrc"
                process_seed = int(init_seed + jobid)
                process = multiprocessing.Process(target=lax.GenerateLaxHandler, args=(dname, ), kwargs={"fname":fname, "seed":process_seed})
                self.jobs.append((process, jobid))
                process.start()
                jobid += 1

            time.sleep(self.status_check)

laxManager = manager("/data/lsiemens/NPDE_C4_large", 2)
laxManager.start()
