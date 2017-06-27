import os
import sys

def NPDE_collect(source, target, recursive=False):
    source_files = []
    data = ""
    for subdir, dirs, files in os.walk(source):
        for file in files:
            file = os.path.abspath(os.path.join(subdir, file))
            if not recursive:
                if os.path.relpath(subdir, source) != ".":
                    continue
            if file.endswith(".dat"):
                source_files.append(file)
    source_files.sort()

    for file in source_files:
        with open(file, "r") as fin:
            data = data + fin.read()
    
    with open(target, "w") as fout:
        fout.write(data)

if __name__ == "__main__":
    recursive = False
    if len(sys.argv) < 3:
        raise RuntimeError("Wrong number of arguments.\nExample: NPDE_collect \"source\" \"target\" [-R]")
    if len(sys.argv) > 3:
        if sys.argv[3] == "-R":
            recursive = True
    NPDE_collect(sys.argv[1], sys.argv[2], recursive)
