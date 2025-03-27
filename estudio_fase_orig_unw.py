#%% 26 Mar NEdd070 
import time
start_time = time.time()
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib as mpl
import pandas as pd
import tkinter as tk
import scipy as sc
import shutil
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.fft import fft, ifft, rfftfreq,irfft
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from sklearn.metrics import r2_score
from pprint import pprint
from tkinter import filedialog
from uncertainties import ufloat, unumpy
from datetime import datetime,timedelta
from numpy.core.numeric import indices
from funciones_procesado import medida_cruda, medida_cruda_autom,ajusta_seno, sinusoide,resta_inter,filtrando_ruido,recorte,promediado_ciclos,fourier_señales_5,lector_templog_2,lector_templog,susceptibilidad_M_0
N_armonicos_impares = 7
def reducir_a_rango_principal_numpy(x):
    '''np.fmod(x + np.pi / 2, np.pi): Calcula el módulo de x+π/2 con respecto a π. 
    Esto asegura que el resultado esté en el intervalo [0,π).'''
    x_mod = np.fmod(x + np.pi / 2, np.pi) # Paso 1: Sumar pi/2 y calcular el módulo pi
    
    x_equiv = x_mod - np.pi / 2 # Paso 2: Restar pi/2 para llevar al rango (-pi/2, pi/2)
      
    return x_equiv

%matplotlib
#%%
path_espectro_NEdd070=os.path.join('original_files','Analisis_20250326_132431','espectros_reconstrucciones','135kHz_100dA_100Mss_NEdd070.txt_Espectro.txt')
#original
data_H = np.loadtxt(path_espectro_NEdd070,skiprows=4,max_rows=1, usecols=(0,1,2))
data_M = np.loadtxt(path_espectro_NEdd070,skiprows=7, usecols=(0,1,2))
f_H,amp_H,fase_H = data_H[0],data_H[1],data_H[2]
f_M = data_M[:, 0]
amp_M = data_M[:, 1]
fase_M = data_M[:, 2]
fase_H_new=np.arctan(np.tan(fase_H))
fase_H_new_new=np.arctan(np.tan(reducir_a_rango_principal_numpy(fase_H)))
fase_M_new=np.arctan(np.tan(fase_M))# arcotangente de la tangente
fase_M_new_new=np.arctan(np.tan(reducir_a_rango_principal_numpy(fase_M)))
#unwraped
path_espectro_NEdd070_unw=os.path.join('original_files','Analisis_20250326_141543','espectros_reconstrucciones','135kHz_100dA_100Mss_NEdd070.txt_Espectro.txt')
data_H_unw = np.loadtxt(path_espectro_NEdd070_unw,skiprows=4,max_rows=1, usecols=(0,1,2))
data_M_unw = np.loadtxt(path_espectro_NEdd070_unw,skiprows=7, usecols=(0,1,2))
f_H_unw,amp_H_unw,fase_H_unw = data_H_unw[0],data_H_unw[1],data_H_unw[2]
f_M_unw = data_M_unw[:, 0]  
amp_M_unw = data_M_unw[:, 1]  
fase_M_unw = data_M_unw[:, 2] 
fase_H_unw_new=np.arctan(np.tan(fase_H_unw))
fase_H_unw_new_new=np.arctan(np.tan(reducir_a_rango_principal_numpy(fase_H_unw)))
fase_M_unw_new=np.arctan(np.tan(fase_M_unw))# arcotangente de la tangente
fase_M_unw_new_new=np.arctan(reducir_a_rango_principal_numpy(fase_M_unw))

fig,(ax,ax2)=plt.subplots(nrows=2,constrained_layout=True,figsize=(8,6),sharex=True)
ax.set_title('Amplitudes',loc='left')
ax.plot(f_M,amp_M,'o-',label='Orig')
ax.plot(f_M_unw,amp_M_unw,'.-',label='Unw')
ax2.set_title('Fases',loc='left')
ax2.plot(f_M,fase_M,'o-',label='Orig')
ax2.plot(f_M,fase_M_new,'.-',label='Orig new')
ax2.plot(f_M,fase_M_new_new,'.-',label='Orig new new')
ax2.plot(f_M_unw,fase_M_unw,'o-',label='Unw')
ax2.plot(f_M_unw,fase_M_unw_new,'.-',label='Unw new')
ax2.plot(f_M_unw,fase_M_unw_new_new,'.-',label='Unw new new')
ax.set_xlabel('f')
ax.grid()
ax2.set_xlabel('H')
ax2.set_ylabel('M')
ax2.grid()
ax.legend(ncol=2)
ax2.legend(ncol=2)
plt.suptitle('Espectro NEdd070 - orig y unw')

#%% Opero con las fases en cada caso
X , X_new, X_new_new = [],[],[]   #X: original, new: arctan(tan(x)) , new new: arctan(tan(rango_ppal(x)))
Y, Y_new, Y_new_new  = [],[],[]
tau,tau_new,tau_new_new=[],[],[]
for n in range(len(f_M)):
    x=(2*n+1)*fase_H-fase_M[n] #n*phi_H - phi_M_n
    X.append(x)
    y=np.tan(x)
    Y.append(y)
    tau.append(y/(2*np.pi*f_M[n]))
    
    x_new=(2*n+1)*fase_H_new-fase_M_new[n]
    X_new.append(x_new)
    y_new=np.tan(x_new)
    Y_new.append(y_new)
    tau_new.append(y_new/(2*np.pi*f_M[n]))
    
    x_new_new=(2*n+1)*fase_H_new_new-fase_M_new_new[n]
    X_new_new.append(x_new_new)
    y_new_new=np.tan(x_new_new)
    Y_new_new.append(y_new_new)
    tau_new_new.append(y_new_new/(2*np.pi*f_M[n]))
    
print('Señal Original')
for n in range(len(f_M)):
    print(f'frec: {f_M[n]:9.0f} Hz |',f'x= {X[n]:7.3f} | ',f'tan(x)= {Y[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y[n]):7.3f} |',f'tau = {tau[n]:5.2e}')
print('\n')
for n in range(len(f_M)):
    print(f'frec: {f_M[n]:9.0f} Hz |',f'x= {X_new[n]:7.3f} | ',f'tan(x)= {Y_new[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y_new[n]):7.3f} |',f'tau = {tau_new[n]:5.2e}')
print('\n')
for n in range(len(f_M)):
    print(f'frec: {f_M[n]:9.0f} Hz |',f'x= {X_new_new[n]:7.3f} | ',f'tan(x)= {Y_new_new[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y_new_new[n]):7.3f} |',f'tau = {tau_new_new[n]:5.2e}')
print('\n')

X=np.array(X)
Y=np.array(Y)
X_new=np.array(X_new)
Y_new=np.array(Y_new)
X_new_new=np.array(X_new_new)
Y_new_new=np.array(Y_new_new)

X_unw = []  
Y_unw = [] 
X_unw_new = []  
Y_unw_new = [] 
X_unw_new_new = []  
Y_unw_new_new = [] 
tau_unw,tau_unw_new,tau_unw_new_new=[],[],[]

for n in range(len(f_M_unw)):
    x_unw=(2*n+1)*fase_H_unw-fase_M_unw[n]
    X_unw.append(x_unw)
    y_unw=np.tan(x_unw)
    Y_unw.append(y_unw)
    tau_unw.append(y_unw/(2*np.pi*f_M_unw[n]))

    x_unw_new=(2*n+1)*fase_H_unw_new-fase_M_unw_new[n]
    X_unw_new.append(x_unw_new)
    y_unw_new=np.tan(x_unw_new)
    Y_unw_new.append(y_unw_new)
    tau_unw_new.append(y_unw_new/(2*np.pi*f_M_unw[n]))

    x_unw_new_new=(2*n+1)*fase_H_unw_new_new-fase_M_unw_new_new[n]
    X_unw_new_new.append(x_unw_new_new)
    y_unw_new_new=np.tan(x_unw_new_new)
    Y_unw_new_new.append(y_unw_new_new)
    tau_unw_new_new.append(y_unw_new_new/(2*np.pi*f_M_unw[n]))
  
print('Señal Unwraped')
for n in range(len(f_M_unw)):
    print(f'frec: {f_M_unw[n]:9.0f} Hz |',f'x= {X_unw[n]:7.3f} | ',f'tan(x)= {Y_unw[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y_unw[n]):7.3f} |',f'tau = {tau_unw[n]:5.2e}')
print('\n')
for n in range(len(f_M_unw)):
    print(f'frec: {f_M_unw[n]:9.0f} Hz |',f'x= {X_unw_new[n]:7.3f} | ',f'tan(x)= {Y_unw_new[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y_unw_new[n]):7.3f} |',f'tau = {tau_unw_new[n]:5.2e}')
print('\n')
for n in range(len(f_M_unw)):
    print(f'frec: {f_M_unw[n]:9.0f} Hz |',f'x= {X_unw_new_new[n]:7.3f} | ',f'tan(x)= {Y_unw_new_new[n]:7.3f} | ',f'arctan(tan(x))= {np.arctan(Y_unw_new_new[n]):7.3f} |',f'tau = {tau_unw_new_new[n]:5.2e}')
print('\n')

X_unw=np.array(X_unw)
Y_unw=np.array(Y_unw)
X_unw_new=np.array(X_unw_new)
Y_unw_new=np.array(Y_unw_new)
X_unw_new_new=np.array(X_unw_new_new)
Y_unw_new_new=np.array(Y_unw_new_new)

#%%
fig,(ax1,ax2)=plt.subplots(nrows=2,constrained_layout=True,figsize=(8,6),sharex=True)

ax1.set_title('n*$\phi_H$ - $\phi_M^n$')

ax1.plot(f_M,X,'o-',label='Orig')
ax1.plot(f_M,X_new,'.-',label='Orig new')
ax1.plot(f_M,X_new_new,'.-',label='Orig new new')

ax1.plot(f_M_unw,X_unw,'o-',label='Unw')
ax1.plot(f_M_unw,X_unw_new,'.-',label='Unw new')
ax1.plot(f_M_unw,X_unw_new_new,'.-',label='Unw new new')

ax1.axhline(np.pi/2,0,1,ls='--',c='k')
# ax1.plot(f_M,np.arctan(Y),'o-',label='arctan tan(Fases)')

ax2.set_title('tan(n*$\phi_H$ - $\phi_M^n$)')
ax2.plot(f_M,Y,'o-',label='Orig')
ax2.plot(f_M,Y_new,'.-',label='Orig new')
ax2.plot(f_M,Y_new_new,'.-',label='Orig new new')

ax2.plot(f_M_unw,Y_unw,'o-',label='Unw')
ax2.plot(f_M_unw,Y_unw_new,'.-',label='Unw new')
ax2.plot(f_M_unw,Y_unw_new_new,'.-',label='Unw new new')
for a in (ax1,ax2):
    a.legend(ncol=2)
    a.grid()
ax2.set_xticks(f_M)    
ax2.set_xlabel('frec')
#%%

fig,(ax,ax2)=plt.subplots(nrows=2,constrained_layout=True,figsize=(8,6),sharex=True)
ax.set_title('Orig')
ax.plot(f_M,fase_M,'o-',label='Orig')
ax.plot(f_M,fase_M_new,'o-',label='arctan tan')
ax.plot(f_M,fase_M_new_new,'o-',label='arctan tan rg ppal')
ax.axhline(np.pi/2,0,1,ls='--',c='k',label='$\pi/2$')

ax2.set_title('Unw')
#ax2.plot(f_M_unw,fase_M_unw,'o-',label='unw')
ax2.plot(f_M_unw,fase_M_unw_new,'o-',label='arctan tan')
ax2.plot(f_M_unw,fase_M_unw_new_new,'o-',label='arctan tan rg ppal')
ax2.axhline(np.pi/2,0,1,ls='--',c='k',label='$\pi/2$')

ax.set_xlabel('f')
ax.grid()

ax2.set_xlabel('H')
ax2.set_ylabel('M')
ax2.grid()
ax.legend()
ax2.legend()
plt.suptitle('n*$\phi_H$ - $\phi_M^n$')
plt.show()
#%%
# fig2,(ax,ax2)=plt.subplots(nrows=2,constrained_layout=True,figsize=(8,6),sharex=True)
# ax.plot(f_M,np.tan(fase_M),'o-',label='tan(Fase) orig')
# ax.plot(f_M,np.tan(fase_M_new),'.-',label='arctan(tan(Fase))')
# ax.plot(f_M,np.tan(fase_M_new_new),'.-',label='tan(arctan(tan(rg ppal Fase)))')
# #ax.axhline(np.pi/2,0,1,ls='--',c='k',label='$\pi/2$')

# ax2.plot(f_M_unw,np.tan(fase_M_unw),'o-',label='tan(fase unw)')
# ax2.plot(f_M_unw,np.tan(fase_M_unw_new),'.-',label='tan(arctan(tan(fase unw))')
# ax2.plot(f_M_unw,np.tan(fase_M_unw_new_new),'.-',label='tan(arctan(tan(fase unw)) rg ppal')

# ax.grid()
# ax2.set_xlabel('frec')
# ax2.grid()
# ax.legend()
# ax2.legend()
# plt.suptitle('tangente de la fase')

# plt.show()



# %%
