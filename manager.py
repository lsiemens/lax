#! /usr/bin/env python3

import lax
import multiprocessing

class manager:
    def __init__(self, fname_base, processes=1):
        self.fname_base = fname_base
        self.processes = processes

        self.jobs = None
        
    def start(self):
        self.jobs = []
        for i in range(self.processes):
            print("starting process: " + str(i))
            fname = self.fname_base + str(i) + ".dat"
            process = multiprocessing.Process(target=lax.GenerateLax, args=(fname, ))
            self.jobs.append(process)
            process.start()

laxManager = manager("TEST_NPDE", 2)
laxManager.start()