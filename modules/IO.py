'''This is a native module for preprocessing of EEG many pipline project using mne python
it consists of several IO functions

# BEHZAD IRAVANI, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# MARCH 2022
'''
# --- Import modules
import numpy as np
# --- End import

def write_text(dirs, fname, line,mode, key):
    # This function writes a line into a text file
    # Input:
    #       dirs  =  a dictionary contains essential paths
    #       fname =  a string contains the name of the text file
    #       line  =  a list contains the lines to be written into the text file
    #       mode  =  a string, 'w' overwrite or 'a' append mode
    #       key   =  a string contains the key value for the "dirs" dictionary

    with open(dirs[key] + "/" + fname + ".txt", mode) as f:
        f.write('\n'.join(line))

def write_organized_events(o):
    # This function can be used to write the events to text file
    # Input:
    #       o  =  a dictionary contains the event data

    report_lines = []
    header = []
    for key, _ in o.items():
        header.append(f'{key}\t')

    report_lines.append(''.join(header))
    report_lines.append('\n')
    arr_event = list(o.values())
    for c in range(len(arr_event)):
        while isinstance(arr_event[c], list):
            arr_event[c] = np.array(arr_event[c]).reshape(-1)

    data2write = (np.array(arr_event)).T
    row, col = data2write.shape
    for r in range(row):
        body = []
        for c in range(col):
            body.append(f'{data2write[r][c]}\t')
        report_lines.append(''.join(body))
    return report_lines

def save_dict(dirs):
    # This function writes dictionary "dirs" to a text file
    # Input:
    #       dirs  =  a dictionary contains the essential paths

    # getting name of dictionary
    with open(dirs["root_folder"] + "\dirs.txt", "w") as f:
        for key, nested in dirs.items():
            if type(nested) is not dict:
                f.write('   {}: {} \n'.format(key, nested))
            else:
                f.write(key + '\n')
                for subkey, value in sorted(nested.items()):
                    f.write('   {}: {} \n'.format(subkey, value))

def read_dirs(fname):
    # This function reads the dictionary "dirs" from the text file
    # Input:
    #       fname  =  a string contains the name of the text file

    keys = []
    values = []
    subkeys = []
    subvalues = []
    next_level = False
    with open(fname + "\dirs.txt") as f:
        while True:
            line = f.readline()[:-1]
            # remove any leading and trailing whitespace
            line = line.strip()
            if not line:
                break
            if ':' in line:

                sline = line.split(':')
                if not next_level:
                    keys.append(sline[0])
                    values.append(sline[1] + ':' + sline[2])
                else:
                    subkeys.append(sline[0])
                    subvalues.append(sline[1] + ':' + sline[2])
            else:
                keys.append(line)
                next_level = True
    dirs = dict()
    for i in range(len(values)):
        dirs[keys[i].strip()] = values[i].strip()
    return dirs
