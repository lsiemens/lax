import os
import sys

class entry:
    def __init__(self, data):
        self.data = data

        self.L = None
        self.A = None
        self.NPDE = None
        self.multiplicative = None
        self.weight = 0.0

        self._process_data()

    def _process_data(self):
        L,A,NPDE = self.data.split("\n", 2)
        self.L, self.A, self.NPDE = len(L), len(A), len(NPDE)
        if NPDE.startswith("+") or NPDE.startswith("-"):
            NPDE = NPDE[1:]
        if ("+" in NPDE) or ("-" in NPDE):
            self.multiplicative = -1.0
        else:
            self.multiplicative = 1.0

class dataFile:
    def __init__(self, fname):
        self.fname = fname
        self.data = []
        
        self._read_file()

    def __str__(self):
        text = ""
        for item in self.data:
            text = text + "PDEs found: [\n" + item.data + "\n]"
        return text

    def write_file(self, fname):
        with open(fname, "w") as fout:
            fout.write(str(self))

    def _read_file(self):
        data = ""
        with open(self.fname, "r") as fin:
            data = fin.read()
        
        data = data.split("\n")
        data_entry = ""
        i = 0
        for line in data:
            if line == "]":
                #EOF found
                break

            if i == 0:
                if ("PDEs found:" in line) or (line == ""):
                    pass
                else:
                    raise IOError("Expected new PDE entry or empty line.")
            elif (i == 1) or (i == 2):
                if line.endswith(";"):
                    data_entry = data_entry + "\n" + line
                else:
                    raise IOError("Expected line terminated with \";\"")
            elif i == 3:
                if line.endswith(","):
                    line = line[:-1]
                data_entry = data_entry + "\n" + line
                self.data.append(entry(data_entry.strip()))
                i = 0
                data_entry = ""
                continue
            i += 1

def NPDE_sort(source, target):
    data = dataFile(source)
    for item in data.data:
        item.weight = item.NPDE*item.multiplicative
    data.data.sort(key=lambda x: x.weight)
    data.write_file(target)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("Wrong number of arguments.\nExample: NPDE_sort \"source\" \"target\"")
    NPDE_sort(sys.argv[1], sys.argv[2])
