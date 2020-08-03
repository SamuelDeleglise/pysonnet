from pysonnet import make_linear_scan
import numpy as np
from matplotlib import pyplot as plt
import os
#  We define the interval over which to scan the memrbane height 

IS_RECALCULATE = True # whether to use saved data or recalculate with SONNET (lengthy...)
HS = np.logspace(-3, 3, 43)
FREQS_PARENT = np.linspace(0.1,20,1001)
FREQS_INDUCTANCE_SINGLE_H = np.concatenate([np.linspace(0.1,100,100),np.linspace(14,20,100)])
FREQS_INDUCTANCE_ALL_H = np.concatenate([np.linspace(0.1,100,100),np.linspace(14,20,100)])
FREQS_CAPA_SINGLE_H = np.linspace(0.1,100,100)
FREQS_CAPA_ALL_H = np.array([0.1])
H_EVALUATE = 1000
DIRNAME = r'C:\Users\Thibault\Documents\phd\sonnet\pysonnet test\BM2'
plt.close('all')
os.chdir(DIRNAME)


PARENT_PROJECT = "B2_OMIT.son"#"A1_OMIT_net.son"

CAPA_PROJECT = "B2_OMIT_s2.son" #PARENT_SUFFIX + 's2' + '.son'

INDUCTOR_PROJECT = "B2_OMIT_s1.son"

#%%


zline_mat_of_h = None#np.zeros((len(HS), NLINE_FREQS, NPORTS_LINE, NPORTS_LINE), dtype=complex)
for index, h in enumerate(HS):
    
    f_line, y = make_linear_scan(PARENT_PROJECT, FREQS_PARENT, h=h)
    if zline_mat_of_h is None:
        zline_mat_of_h=np.zeros((len(HS), *y.shape), dtype=complex)
    zline_mat_of_h[index] = y
np.save("zline_mat_of_h.npy", zline_mat_of_h)       
np.save('f_line.npy', f_line)
np.save('HS.npy', HS)

#%%
florentz_of_h = []    
for index, h in enumerate(HS):
    plt.plot(f_line[:-1], np.diff(np.angle(zline_mat_of_h[index, :,0,0])))
    plt.title("membrane height h=%.3f"%h)
    florentz_of_h.append(f_line[np.argmax(np.diff(np.angle(zline_mat_of_h[index, :,0,0])))])
florentz_of_h = np.array(florentz_of_h)



plt.figure()
plt.loglog(HS, florentz_of_h, 'o-')

plt.show()
#%%
### Now, let's focus on the last point (h=1000.) and look at the impedance matrix of individual elements:

f_i_single_h, zi_mat_single_h = make_linear_scan(INDUCTOR_PROJECT,FREQS_INDUCTANCE_SINGLE_H, h=H_EVALUATE)
f_c_single_h, zc_mat_single_h = make_linear_scan(CAPA_PROJECT, FREQS_CAPA_SINGLE_H, h=H_EVALUATE)
np.save('zi_mat_single_h.npy', zi_mat_single_h)
np.save('zc_mat_single_h.npy', zc_mat_single_h)
np.save('f_i_single_h.npy', f_i_single_h)
np.save('f_c_single_h.npy', f_c_single_h)
np.save('h_evaluate.npy', np.array([H_EVALUATE]))


#%%


zi_mat_of_h = np.zeros((len(HS), len(FREQS_INDUCTANCE_ALL_H), 4, 4), dtype=complex)
zc_mat_of_h = np.zeros((len(HS), len(FREQS_CAPA_ALL_H), 2, 2), dtype=complex)
for index, h in enumerate(HS):
    f_zi_of_h, zi_mat = make_linear_scan(INDUCTOR_PROJECT, FREQS_INDUCTANCE_ALL_H, h=h)
    f_zc_of_h, zc_mat = make_linear_scan(CAPA_PROJECT, FREQS_CAPA_ALL_H, h=h)
    zi_mat_of_h[index] = zi_mat
    zc_mat_of_h[index] = zc_mat
np.save('zi_mat_of_h.npy', zi_mat_of_h)
np.save('f_zi_of_h.npy', f_zi_of_h)
np.save('zc_mat_of_h.npy', zc_mat_of_h)
np.save('f_zc_of_h.npy', f_zc_of_h)

#%%
def zeq(z, port1=-1, port2=-2):
    return z[...,port1,port1] + z[...,port2,port2] - \
                         z[...,port1,port2] - z[...,port2,port1]
