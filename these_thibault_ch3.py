from pysonnet import make_linear_scan
import numpy as np


#  We define the interval over which to scan the memrbane height 
hs = np.logspace(-3, 3, 43)
#%%

# We perform a naive scan of the full project (looking via feedline ports and finding a lorentzian response)

plt.figure()
f0s = []
for h in hs:
    x, y = make_linear_scan("A1_OMIT_net_further.son", 4.5 , 6., 1001, h=h)
    plt.plot(x[:-1], np.diff(np.angle(y[:,1,0])))
    f0s.append(x[np.argmax(np.diff(np.angle(y[:,1,0])))])

np.save('f0s_further.npy', f0s)
np.save('hs_further.npy', hs)
#%%
plt.figure()
plt.semilogx(hs, f0s, 'o-')

plt.show()
#%%
# Now, we will try to reproduce these results by looking at the impedance matrix of the capacitor and inductor subsystems only.
# One gotcha, ni order to retrieve the asme frequencies, is to make sure both 
# subprojects have the "not de-embeded" option selected in the "output file" menu.
# Also, make sure the output file is configured to Z, real-imag

xs = []
ys = []

zc = np.zeros((len(hs), 2, 2), dtype=complex)
zi = np.zeros((len(hs), 4, 4), dtype=complex)

for i, h in enumerate(hs):
    x, y = make_linear_scan("A1_OMIT_net_s2.son", 5.7, 5.7, 1, h=h) # capacitance
    # only calculate at 6 GHz
    for index in range(len(x)): # careful, all existing data are returned--> postselect
        if x[index]==5.7:
            zc[i] = y[index]
    x, y = make_linear_scan("A1_OMIT_net_s1_further.son", 5.7, 5.7, 1, h=h) # inductance
    for index in range(len(x)): # postselect frequency
        if x[index]==5.7:
            zi[i] = y[index]
    
#%%
# calculate capacitance and inductance using my formula.
# to demonstrate it, we can start with the defintion of the impedance matrix: 
#            Vxi = sum_j(Zxji Ixi) where x is either i for inductor, or c for capacitor
# By symmetry + port connections, we can write 
#Ic1 = -Ic2 = -Ii1 = Ii2 = I 
#Vc1 = Vi1 = - Vc2 = -Vi2 = V
            
# So, we get Vx1 - Vx2 = (Zx11 + Zx22 - Zx21 - Zx12) I
# Now, if we assume that the inductor behaves as an inductor:
# Vi1 - Vi2 = 2 V = j L omega --->  L = im(Zi11 + Zi22 - Zi21 - Zi12)/omega
# If the capacitor behaves as a capacitor:
# Vc1 - Vc2 = 2 V = 1/j C omega  ---> 1/ omega im(Zc11 + Zc22 - Zc21 - Zc12)


inductance = 1e9*np.imag(zi[:,2,2] + zi[:,3,3] - zi[:,2,3] - zi[:,3,2])/(2*np.pi*5.7e9)
capas = -1.e12/(2*np.pi*5.7e9*np.imag(zc[:,0,0] + zc[:,1,1] - zc[:,0,1] - zc[:,1,0]))

### I leave for reference the sonnet definition of capa and inductance (one of them)
### I have no clue why it doesn't match mine...
#capas = -1.0E12 / (2*np.pi * 6e9 * np.imag((zc[:,0,0] * zc[:,1,1] - zc[:,0,1] * zc[:,1,0]) / zc[:,1,0]))
#inductance = 1e9 * np.imag((zi[:,2,2] * zi[:,3,3] - zi[:,2,3] * zi[:,3,2]) / zi[:,3,2]) / (2*np.pi * 6e9)

#%%
from pylab import *


# plot inductance and capacitance as a function of membrane distance in the 
#first subplot
ax1 = plt.subplot(211)

color = 'tab:red'
ax1.set_xlabel(r'Membrane height ($\mu$m)')
ax1.set_ylabel('Capacitance (pF)', color=color)
ax1.semilogx(hs, capas, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Inductance (nH)', color=color)  # we already handled the x-label with ax1
ax2.semilogx(hs, inductance, color=color)
ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()

#plt.semilogx(hs, inductance, label="Inductance (nH)")
#plt.semilogx(hs, capas, label="Capacitance (pF)")



ax3 = plt.subplot(212, sharex=ax1)

lc_freqs = 1e-9/(2*np.pi*sqrt(inductance*1e-9*capas*1e-12))

plt.semilogx(hs, lc_freqs, color='red')
plt.ylabel("Resonance freq. (GHz)")
plt.xlabel(r'Membrane height ($\mu$m)')

plt.xlim((1e-3, 1e3))

DEBUG_RESONANCE_LORENTZ = False
if DEBUG_RESONANCE_LORENTZ: # in practice, there is still a small gap between 
    # what we get from the lorentzian and what we get from calculating 1/sqrt(LC)
    # I am not sure where this comes from (maybe setting the feedline at 50 ohms instead of open ?)
    # I suggest for now, we just hide the Lorentzian one
    hs = np.load("hs_further.npy")
    f0s = np.load("f0s_further.npy")
    plt.semilogx(hs, f0s)


## Now, let's try to fit with the analytic formula

from scipy.constants import epsilon_0
from scipy.optimize import leastsq

eps_si = 11.9
eps_sin = 7.5
r_si = (1 - eps_si)/(1 + eps_si)
r = (1 - eps_sin)/(1 + eps_sin)
t = 2/(1 + eps_sin)
rb = (eps_sin-1)/(1 + eps_sin)
tb = 2*eps_sin/(1 + eps_sin)
a = 0.5e-6
thick = 200e-9

expo = exp(-2*thick*pi/a)
r_membrane = r + t*tb*rb*expo/(1 + rb**2*expo)

STOP = -15

h = np.logspace(-3,3,43)*1e-6
#h = h[:STOP]
expoh = exp(-2*h*pi/a)
membrane_area = 45e-6*80e-6
Cinf = epsilon_0*(1+eps_si)/(2*a)*membrane_area
C = Cinf*(1 - r_si*r_membrane*expoh)/(1 + r_membrane*expoh)

def freq(L, C0):
    return 1/(2*np.pi*sqrt(L*(C + C0)))

def to_minimize_freq(args):
    L, C0 = args
    return freq(L, C0) - lc_freqs[:STOP]*1e9

def to_minimize_capa(args):
    C0 = args[0]
    return capas - C - C0


pars, flag = leastsq(to_minimize_capa, [1e-9])

ax1.semilogx(h*1e6, C*1e12, '--', color='red')
ax3.semilogy(h*1e6, 1.e-9/(2*pi*np.sqrt(inductance*1e-9*C)), '--', color='red')

ax3.set_yticks([4.5, 5.0, 5.5])
ax3.set_yticklabels(['4.5', '5.0', '5.5'])

ax1.set_yticks([0.4,0.5,0.6])
#plt.semilogx(h*1e6, freq(*pars)*1e-9, '--', color = 'blue')
#plt.semilogx(h*1e6, lc_freqs[:STOP])
ax3.yaxis.set_label("")

#%%
#hs = np.load("newnew_hs.npy")
#f0s = np.load("newnew_f0s.npy")
#plt.semilogx(hs, f0s)
