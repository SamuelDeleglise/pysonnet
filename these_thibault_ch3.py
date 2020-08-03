from pysonnet import make_linear_scan
import numpy as np
from scipy.optimize import leastsq
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sqrt
import os
#  We define the interval over which to scan the memrbane height 

IS_RECALCULATE = True # whether to use saved data or recalculate with SONNET (lengthy...)

plt.close('all')
os.chdir("G:\Mon Drive\Reviews\These_thibault\sonnet")

#%%
PARENT_PROJECT = "Resonator_circle_net.son"#"A1_OMIT_net.son"
PARENT_SUFFIX = PARENT_PROJECT.rstrip('.son') + '_'

CAPA_PROJECT = "Resonator_pad_net_s2.son" #PARENT_SUFFIX + 's2' + '.son'
CAPA_SUFFIX = CAPA_PROJECT.rstrip(".son") + '_'

INDUCTOR_PROJECT = "circle_s1.son"
INDUCTOR_SUFFIX = INDUCTOR_PROJECT.rstrip(".son") + '_'

NLINE_FREQS = 1001 #1001
NPORTS_LINE = 1
FREQ_START_LINE = 0.1
FREQ_STOP_LINE = 15

#%%
PARENT_PROJECT = "B2_OMIT.son"#"A1_OMIT_net.son"
PARENT_SUFFIX = PARENT_PROJECT.rstrip('.son') + '_'

CAPA_PROJECT = "B2_OMIT_s2.son" #PARENT_SUFFIX + 's2' + '.son'
CAPA_SUFFIX = CAPA_PROJECT.rstrip(".son") + '_'

INDUCTOR_PROJECT = "B2_OMIT_s1.son"
INDUCTOR_SUFFIX = INDUCTOR_PROJECT.rstrip(".son") + '_'

NLINE_FREQS = 1001 #1001
NPORTS_LINE = 2
FREQ_START_LINE = 1
FREQ_STOP_LINE = 15

#%%
PARENT_PROJECT = "A1_OMIT_net.son"
PARENT_SUFFIX = PARENT_PROJECT.rstrip('.son') + '_'

CAPA_PROJECT = 'A1_OMIT_net_s2.son'
CAPA_SUFFIX = CAPA_PROJECT.rstrip(".son") + '_'

INDUCTOR_PROJECT = 'A1_OMIT_net_s1.son'
INDUCTOR_SUFFIX = INDUCTOR_PROJECT.rstrip(".son") + '_'

NLINE_FREQS = 1001 #1001
NPORTS_LINE = 2
FREQ_START_LINE = 4.5
FREQ_STOP_LINE = 8.



#%%

# We perform a naive scan of the full project (looking via feedline ports and finding a lorentzian response)

plt.figure()



if IS_RECALCULATE:
    hs = np.logspace(-3, 3, 2)#43
    zline_mat_of_h = np.zeros((len(hs), NLINE_FREQS, NPORTS_LINE, NPORTS_LINE), dtype=complex)
    for index, h in enumerate(hs):
        f_line, y = make_linear_scan(PARENT_PROJECT, FREQ_START_LINE , FREQ_STOP_LINE, NLINE_FREQS, h=h)
        zline_mat_of_h[index] = y
    np.save(PARENT_SUFFIX + "zline_mat_of_h.npy", zline_mat_of_h)       
    np.save(PARENT_SUFFIX + 'f_line.npy', f_line)
    np.save(PARENT_SUFFIX + 'hs.npy', hs)
else:
    hs = np.logspace(-3, 3, 43)
    hs = np.load(PARENT_SUFFIX + 'hs.npy')
    f_line = np.load(PARENT_SUFFIX + 'f_line.npy') 
    zline_mat_of_h = np.load(PARENT_SUFFIX + "zline_mat_of_h.npy")
    NLINE_FREQS = len(f_line) #1001
    NPORTS_LINE = zline_mat_of_h.shape[-1]
    FREQ_START_LINE = f_line[0]
    FREQ_STOP_LINE = f_line[-1]

florentz_of_h = []    
for index, h in enumerate(hs):
    plt.plot(f_line[:-1], np.diff(np.angle(zline_mat_of_h[index, :,0,0])))
    plt.title("membrane height h=%.3f"%h)
    florentz_of_h.append(f_line[np.argmax(np.diff(np.angle(zline_mat_of_h[index, :,0,0])))])
florentz_of_h = np.array(florentz_of_h)


#%%
plt.figure()
plt.loglog(hs, florentz_of_h, 'o-')

plt.show()
#%%
### Now, let's focus on the last point (h=1000.) and look at the impedance matrix of individual elements:
h = 1000
if IS_RECALCULATE:
    f_line, zline_mat_h1000 = make_linear_scan(PARENT_PROJECT, FREQ_START_LINE, FREQ_STOP_LINE, 1001, h=h)
    f_h1000, zi_mat_h1000 = make_linear_scan(INDUCTOR_PROJECT, 1., 100, 100, h=h)
    f_h1000, zc_mat_h1000 = make_linear_scan(CAPA_PROJECT, 1., 100, 100, h=h)
    #zi_mat_h1000 = zi_mat_h1000
    np.save(INDUCTOR_SUFFIX + 'zi_mat_h1000.npy', zi_mat_h1000)
    np.save(CAPA_SUFFIX + 'zc_mat_h1000.npy', zc_mat_h1000)
    np.save(PARENT_SUFFIX + 'zline_mat_h1000.npy', zline_mat_h1000)
    np.save(PARENT_SUFFIX + 'f_h1000.npy', f_h1000)
    np.save(PARENT_SUFFIX + 'f_line.npy', f_line)
else:
    zi_mat_h1000 = np.load(INDUCTOR_SUFFIX + 'zi_mat_h1000.npy')
    zc_mat_h1000 = np.load(CAPA_SUFFIX + 'zc_mat_h1000.npy')
    zline_mat_h1000 = np.load(PARENT_SUFFIX + 'zline_mat_h1000.npy')
    f_h1000 = np.load(PARENT_SUFFIX + 'f_h1000.npy')
    f_line = np.load(PARENT_SUFFIX + 'f_line.npy') 




#%%
# Now let's calculate the impedance of the circuit elements

def zsym(z, port1=-1, port2=-2):
    """
    calculate the impedance for a symmetric mode:
        we can start with the defintion of the impedance matrix: 
            Vxi = sum_j(Zxji Ixi) 
        where x is either i for inductor, or c for capacitor.
        By symmetry + port connections, we can write 
            Ic1 = -Ic2 = -Ii1 = Ii2 = I 
            Vc1 = Vi1 = - Vc2 = -Vi2 = V          
        So, we get Vx1 - Vx2 = (Zx11 + Zx22 - Zx21 - Zx12) I
        The quantity in the parentheses of the rhs is what is calculated here.
        
        For instance, if we assume that the inductor behaves as an inductor:
        Vi1 - Vi2 = 2 V = j L omega --->  L = im(Zi11 + Zi22 - Zi21 - Zi12)/omega
        If the capacitor behaves as a capacitor:
        Vc1 - Vc2 = 2 V = 1/j C omega  ---> 1/ omega im(Zc11 + Zc22 - Zc21 - Zc12)
        
    """
    return np.imag(z[..., port1, port1] - z[..., port1, port2] + z[..., port2, port2] - z[..., port2, port1])
    
zi_h1000 = zsym(zi_mat_h1000)
zc_h1000 = zsym(zc_mat_h1000)

plt.figure()
ax = plt.subplot(211)
plt.ylabel("waveguide response")
plt.plot(f_line, np.imag(zline_mat_h1000[:,0,0]))
plt.subplot(212, sharex=ax)
plt.plot(f_h1000, zi_h1000 + zc_h1000)
plt.hlines([0], 1, 100)
plt.ylim(-1,1)
plt.xlim(min(f_line), max(f_line))
plt.ylabel("zi + zc")
plt.xlabel("freq (GHz)")
###inductance = 1e9*np.imag(zi[:,2,2] + zi[:,3,3] - zi[:,2,3] - zi[:,3,2])/(2*np.pi*5.7e9)
###capas = -1.e12/(2*np.pi*5.7e9*np.imag(zc[:,0,0] + zc[:,1,1] - zc[:,0,1] - zc[:,1,0]))

### I leave for reference the sonnet definition of capa and inductance (one of them)
### I have no clue why it doesn't match mine...
#capas = -1.0E12 / (2*np.pi * 6e9 * np.imag((zc[:,0,0] * zc[:,1,1] - zc[:,0,1] * zc[:,1,0]) / zc[:,1,0]))
#inductance = 1e9 * np.imag((zi[:,2,2] * zi[:,3,3] - zi[:,2,3] * zi[:,3,2]) / zi[:,3,2]) / (2*np.pi * 6e9)

#%%
### Now let's try to fit the responses in frequency domain

def zmodel(args):
    L, f0 = args
    omega0 = 2*pi*f0*1e9
    omega = 2*pi*f_h1000*1e9
    return L*omega/(1 - (omega/omega0)**2)

def zmodel_naive(args):
    L, = args
    omega = 2*pi*f_h1000*1e9
    return L*omega

    
def to_minimize(args, zi):
    return np.abs(zi - zmodel(args))/f_h1000**5# np.log(np.abs(zi - zmodel(args)))

def to_minimize_naive(args, zi):
    return np.abs(zi - zmodel_naive(args))[:10]

def guess_model(zi):
    f0 = f_h1000[np.where(np.diff(zi)<0)[0][0]] + .1
    L = 2e-9
    return L, f0 


res, flag = leastsq(to_minimize, guess_model(zi_h1000), zi_h1000)
L, f0 = res


res_naive, flag = leastsq(to_minimize_naive, [2e-9], zi_h1000)

plt.figure()
ax = plt.subplot(211)
plt.plot(f_h1000, zi_h1000)
plt.plot(f_h1000, zmodel(res))
plt.plot(f_h1000, zmodel_naive(res_naive), "--")


def zmodel_c(args):
    C, = args
    omega = 2*pi*f_h1000*1e9
    return -1./(C*omega)

def to_minimize_c(args, zc):
    return np.abs(zc - zmodel_c(args))[:20]

res, flag = leastsq(to_minimize_c, [1e-12], zc_h1000)
C, = res

plt.subplot(212, sharex=ax)
plt.plot(f_h1000, zc_h1000)
plt.plot(f_h1000, zmodel_c(res))



f_lc = 1e-9/sqrt(L*C)/(2*pi)
f_predicted = f0*f_lc/sqrt(f0**2 + f_lc*2)


#%%
if IS_RECALCULATE:
    N_F_ZI = 100
    N_F_ZC = 1
    zi_mat_of_h = np.zeros((len(hs), N_F_ZI, 4, 4), dtype=complex)
    zc_mat_of_h = np.zeros((len(hs), N_F_ZC, 2, 2), dtype=complex)
    for index, h in enumerate(hs):
        f_zi_of_h, zi_mat = make_linear_scan(INDUCTOR_PROJECT, 1., 100, N_F_ZI, h=h)
        f_zc_of_h, zc_mat = make_linear_scan(CAPA_PROJECT, 5., 5., N_F_ZC, h=h)
        zi_mat_of_h[index] = zi_mat
        zc_mat_of_h[index] = zc_mat
    np.save(INDUCTOR_SUFFIX + 'zi_mat_of_h.npy', zi_mat_of_h)
    np.save(INDUCTOR_SUFFIX + 'f_zi_of_h.npy', f_zi_of_h)
    np.save(CAPA_SUFFIX + 'zc_mat_of_h.npy', zc_mat_of_h)
    np.save(CAPA_SUFFIX + 'f_zc_of_h.npy', f_zc_of_h)
else:
    zi_mat_of_h = np.load(INDUCTOR_SUFFIX + 'zi_mat_of_h.npy')
    f_zi_of_h = np.load(INDUCTOR_SUFFIX + 'f_zi_of_h.npy')
    zc_mat_of_h = np.load(CAPA_SUFFIX + 'zc_mat_of_h.npy')
    f_zc_of_h = np.load(CAPA_SUFFIX + 'f_zc_of_h.npy')
    
#%%

zi_of_h = zsym(zi_mat_of_h, 2,3)
zc_of_h = zsym(zc_mat_of_h)

Ls = np.zeros(len(hs)) # DC inductance of the inductor
f0s = np.zeros(len(hs)) # self resonance of the inductor

Cs = np.zeros(len(hs)) # capacitance of the capacitor

plt.close('all')
IS_PLOT_FITS = False

for index,(zi, zc) in enumerate(zip(zi_of_h, zc_of_h)):
    
    # fit inductor parameters
    res, flag = leastsq(to_minimize, guess_model(zi), zi)
    L, f0 = res
    Ls[index] = L
    f0s[index] = f0
    if IS_PLOT_FITS:
        plt.figure()
        plt.plot(f_zi_of_h, zi, 'o')
        plt.plot(f_zi_of_h, zmodel(res))
    
    # fit capacitor parameter
    C = (-1./(zc*2*pi*f_zc_of_h*1e9)).mean()
    Cs[index] = C
ax = plt.subplot(411)
plt.semilogx(hs, Ls)
plt.subplot(412, sharex=ax)
plt.semilogx(hs, f0s)
plt.subplot(413, sharex=ax)
plt.semilogx(hs, Cs)
plt.subplot(414, sharex=ax)
f_lc = 1.e-9/(2*pi*sqrt(Ls*Cs))
predicted_freq = f_lc * f0s/sqrt(f_lc**2 + f0s**2)
plt.semilogx(hs, f_lc)
plt.semilogx(hs, predicted_freq)
plt.semilogx(hs, florentz_of_h, 'o')



