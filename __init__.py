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

import os
from numpy import linspace
import numpy as np
from subprocess import call, SubprocessError, Popen
from glob import glob

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

def filter_output(x, y, points):
    """
    remove points in x, y that are absent in points.
    Assumes all points are in x
    """
    
    filtered_x = []
    filtered_y = []
    
    index = 0
    next_freq = points[index]
    for xx, yy in zip(x,y):
        if np.abs(xx - next_freq)<1e-8:
            filtered_x.append(xx)
            filtered_y.append(yy)
            index+=1
            if index>=len(points):
                break
            else:
                next_freq = points[index]
    return np.array(filtered_x), np.array(filtered_y)

def make_linear_scan(project, start, stop, npoints, **params):
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
    # In order for python to read out the data,
    # The sonnet project should define a "spreadsheat type output_file" (analysis->output_file)
    # The name should simply be project_name.csv in the project folder
    # if params['two_ports']:
    resfiles = (project.replace('.son', '.s1p'), project.replace('.son', '.s2p'))
    #glob(project.replace('.son', '.s*p'))[0]  # '.csv' #if working with two ports, use '.s2p' instead
    # else:
    #    resfile = project.replace('.son', '.s1p')
   
    for resfile in resfiles:
        if os.path.exists(resfile):
            os.remove(resfile)

    # Define the frequencies of the scan by creating a file freqs.eff in the current directory
    points = linspace(start, stop, npoints)
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
    from subprocess import check_call, check_output
    error = False
    with open('err.txt', 'w') as err:
        try:
            # C:\\Program Files (x86)\\Sonnet Software\\13.52\\bin_x64\\
            p = Popen(
                ["em.exe", "-ParamFile", 'params.txt', project, "freqs.eff"],
                stderr=err)
            error = p.wait()
        except SubprocessError as e:
            error = True
        finally:
            print("terminate", error)
            p.terminate()
    if error:  # Output the console error if any
        with open('err.txt', 'r') as err:
            raise SubprocessError(err.read())


    resfile = glob(project.replace('.son', '.s*p'))[0]
    x, y = read_output(resfile)
    return filter_output(x, y, points)
    