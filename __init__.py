"""
Calls sonnet solver on the required project and performs a linear frequency scan
(an external frequency file is used for more flexibility). Extra parameters given
as keyword arguments are overwritten in the project (using also an external parameter file).

Example: make_linear_scan(PROJECT, 6., 7., 10000, altitude=0.15)

GOTCHAS:
- If the project requires subprojects, uncheck the "hierarchy scan" in analysis->output_file
as interpolation will provide a huge speedup
- The subproject should have a reasonable scan range for the parent project to be able to
interpolate at the right frequencies: typically, if you plan to analyse the parent project
from 6 GHz to 7 GHz, in each subproject, go to analysis->setup and choose Linear Scan
from 6. to 7. GHz by steps of .1 GHz (irrespective of the large number of points you may ask
on the parent scan)
"""

import os, sys
from numpy import linspace
import numpy as np
from subprocess import SubprocessError, Popen

def read_matrix(file):
    """
    assumes the cursor is at the beginning of a n x m matrix.
    Reads new line as long as space is present at the beginning of the lines

    returns a numpy array
    """
    lines = ''
    #nlines = 0
    while(True):
        pos = file.tell()
        line = file.readline()
        if line.startswith(' '):
            lines += line.strip() + '\n'
            #nlines += 1
        else:
            file.seek(pos)
            break
    mat = np.fromstring(lines, sep=' ')
    #if nlines>1:
    n2 = mat.shape[0]//2
    n = int(np.sqrt(n2))
    mat = mat.reshape((n, 2*n))
    return mat

def read_freq(file):
    pos = file.tell()
    line = file.readline()
    if line == '':
        raise EOFError("")
    string = line.split(' ')[0]
    file.seek(pos + len(string))
    return float(string)


def read_comments(file):
    pos = file.tell()
    while (True):
        line = file.readline()
        if line == '':
            raise EOFError("")
        if line.startswith('!'):
            pos = file.tell()
        else:
            file.seek(pos)
            break


def read_output(filename):
    with open(filename, 'r') as f:
        while (not f.readline().startswith("#")):
            pass
        freqs = []
        mats = []
        while (True):
            try:
                freqs.append(read_freq(f))
                mats.append(read_matrix(f).view(complex))
                read_comments(f)
            except EOFError:
                return np.array(freqs), np.array(mats)

def get_output_config(filename): #Search if we are dealing with a .s2p or .s4p output
    with open(filename, 'r') as f:
        found=False
        for line in f.readlines():
            if found:
                fileout=line
                break
            elif line.startswith('FILEOUT'):
                found=True
    return fileout[fileout.find('$BASENAME')+9:fileout.find('$BASENAME')+13]

def make_linear_scan(*args, **params):
    """
    Calls sonnet solver on the required project and performs a linear frequency scan
    (an external frequency file is used for more flexibility). Extra parameters given
    as keyword arguments are overwritten in the project (using also an external parameter file).
    
    either specify the start, stop and Npoint frequency, either specify the list of freqs 
    
    Example: -make_linear_scan(PROJECT, 6., 7., 10000, altitude=0.15)
             -make_linear_scan(PROJECT, [6,6.4,6.8], altitude=0.15)

    GOTCHAS:
    - If the project requires subprojects, uncheck the "hierarchy scan" in analysis->output_file
    as interpolation will provide a huge speedup
    - The subproject should have a reasonable scan range for the parent project to be able to
    interpolate at the right frequencies: typically, if you plan to analyse the parent project
    from 6 GHz to 7 GHz, in each subproject, go to analysis->setup and choose Linear Scan
    from 6. to 7. GHz by steps of .1 GHz (irrespective of the large number of points you may ask
    on the parent scan)
    """
    # In order for python to read out the data,
    # The sonnet project should define a "spreadsheat type output_file" (analysis->output_file)
    # The name should simply be project_name.csv in the project folder
    # if params['two_ports']:
    if len(args)==4:
        project, start, stop, npoints=args
        points = linspace(start, stop, npoints)
    elif len(args)==2:
        project, points=args
    dirname= os.path.dirname(project)
    previous_dir=os.getcwd()
    if dirname!='':
        os.chdir(dirname)
    project=os.path.split(project)[-1]
    extension=get_output_config(project)
    resfile = project.replace('.son',
                              extension)  # '.csv' #if working with two ports, use '.s2p' instead
    # else:
    #    resfile = project.replace('.son', '.s1p')
    if os.path.exists(resfile):
        os.remove(resfile)

    # Define the frequencies of the scan by creating a file freqs.eff in the current directory
    
    with open('freqs.eff', 'w') as f:
        f.write('FREQUNIT GHZ\n')
        for point in points:
            f.write('STEP ' + str(point) + '\n')

    # define the parameters to use by creating a file params.txt in the current directory
    with open("params.txt", 'w') as f:
        for key, val in params.items():
            f.write(str(key) + '=' + str(val) + '\n')

    # call the Sonnet solver executable with the right command line arguments to take into account
    # - The required project
    # - The external frequency file
    # - The parameters to overwrite
    error = False
    with open('err.txt', 'w') as err:
        try:
            p = Popen(
                ["em.exe", "-ParamFile", 'params.txt', project, 'freqs.eff'],
                stderr=err)
            error = p.wait()
        except SubprocessError as e:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print(os.listdir(os.getcwd()))
            print(project)
            error = True
        finally:
            print("terminate", error)
            p.terminate()
    if error:  # Output the console error if any
        with open('err.txt', 'r') as err:
            raise SubprocessError(err.read())
    res=read_output(resfile)
    os.chdir(previous_dir)
    return res
