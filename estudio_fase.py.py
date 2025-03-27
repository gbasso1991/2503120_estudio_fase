#%% 19 Mar 25 Ciclo simulado Pedro
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
%matplotlib

#%%
archivo = 'datos_dos_ciclos.txt'
# Leer el archivo y cargar los datos en un array de NumPy
data = np.loadtxt(archivo)

# Separar las columnas en tres arrays diferentes
t = data[:, 0]  # Primera columna
H = data[:, 1]  # Segunda columna
M = data[:, 2]  # Tercera columna

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(t,H/max(H))
ax.plot(t,M/max(M))
ax.set_xlabel('t')
ax.set_title('H',loc='left')
ax.set_title('M',loc='left')
ax.grid()
plt.xlim(0,t[-1])
#plt.savefig('señal_cruda.png')
plt.show()
plt.plot(H,M)
# t=t-t[0]

offset , amp, frec , fase = ajusta_seno(t,H)
print(offset , amp, frec , fase)

y=sinusoide(t,offset , amp, frec , fase)
print('FFT del script mio')
_, _, muestra_rec_impar,delta_phi_0,f_0,amp_0,fase_0, espectro_f_amp_fase_m,espectro_ref = fourier_señales_5(t,M,H,
                                                                                                        delta_t=2e-8,polaridad=1,
                                                                                                        filtro=0.05,frec_limite=13*frec,
                                                                                                        name='test')
plt.close('all')
# fig_fourier.savefig(os.path.join(os.getcwd(),'simulado_Espectro.png'),dpi=200,facecolor='w')
# fig2_fourier.savefig(os.path.join(os.getcwd(),'simulado_Rec_impar.png'),dpi=200,facecolor='w')

#FASE
frecs_M=espectro_f_amp_fase_m[0]
frecs_H=espectro_ref[0]

Fase_H=espectro_ref[2]
Fase_M= espectro_f_amp_fase_m[2]

print('''Espectro de la señal de Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
print(f"{frecs_H:.8e} {espectro_ref[1]:.8e} {Fase_H:.8e}")

print('''\nEspectro de la señal de muestra:\nFrecuencia_Hz|Intensidad rel|Dphi''')
for i in range(len(frecs_M)):
    print(f'{frecs_M[i]:.8e} {espectro_f_amp_fase_m[1][i]:.8e} {Fase_M[i]:.8e}')
 
tau_0=np.tan(Fase_H-Fase_M[0])/(2*np.pi*frecs_M[0])
print('\ntau (1er armonico)= tan(Fase_H - phi0_M)/(2*pi*f0) = ',f'{tau_0:.2e} s\n')

# %% Resultados Pedro
fft_M=np.array([ 4.43278870e-01-1.72955778e-01j,
        -6.98353434e-02+1.74327921e-02j,
          2.45573697e-02+5.84966660e-03j,
         -8.50069276e-03-8.62516981e-03j,
         1.29000184e-03+6.69786628e-03j])

fft_H=np.array([  7.51402967e+06+2.43102125e+06j,
        -6.38293117e-11-1.78918811e-10j,
         3.20291789e-12-8.80074434e-13j,
       -6.78002834e-11-1.45286322e-10j,
        -5.17772851e-11-7.70143130e-12j])

fases_M_Pedro = np.angle(fft_M)
fases_H_Pedro = np.angle(fft_H)

mag_M_Pedro = np.abs(fft_M)
mag_H_Pedro = np.abs(fft_H)

frecs_aux=np.arange(1e4,11e4,2e4)
print('Resultados Pedro')
print('''Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
print(f"{frecs_aux[0]:.8e} {mag_H_Pedro[0]:.8e} {fases_H_Pedro[0]:.8e}")
print('''\nMagnetizacion\nFrecuencia_Hz|Intensidad rel|Dphi''')
for i in range(len(frecs_aux)):
    print(f'{frecs_aux[i]:.8e} {mag_H_Pedro[i]:.8e} {fases_M_Pedro[i]:.8e}')

tau_0_Pedro=np.tan(fases_H_Pedro[0]-fases_M_Pedro[0])/(2*np.pi*frecs_aux[0])
print('\ntau (1er armonico)= tan(Fase_H - phi0_M)/(2*pi*f0) = ',f'{tau_0_Pedro:.2e} s\n')

#%% Veo como da Tau, tg etc

X = [] #argumento de la tg
Y = [] #tangente 
for n in range(len(frecs_M)):
    x=(2*n+1)*Fase_H-Fase_M[n]
    X.append(x)
    
    y=np.tan(x)
    Y.append(y)
    print(f'frec: {frecs_M[n]:9.0f} Hz -',f'x= {x:5.3f} - ',f'tan(x)= {y:6.3f} - ',f'arctan(tan(x))= {np.arctan(y):5.3f}')
X=np.array(X)
Y=np.array(Y)

#La operacion de tg y arcotangente te lo manda al rango necesario
def reducir_a_rango_principal_numpy(x):
    '''np.fmod(x + np.pi / 2, np.pi): Calcula el módulo de x+π/2 con respecto a π. 
    Esto asegura que el resultado esté en el intervalo [0,π).'''
    x_mod = np.fmod(x + np.pi / 2, np.pi) # Paso 1: Sumar pi/2 y calcular el módulo pi
    
    x_equiv = x_mod - np.pi / 2 # Paso 2: Restar pi/2 para llevar al rango (-pi/2, pi/2)
      
    return x_equiv
    
#print(X)
X_equiv = reducir_a_rango_principal_numpy(X)
#print(X_equiv)

fig,ax=plt.subplots(nrows=2,constrained_layout=True,sharex=True)
ax[0].set_title('n*$\phi_H$ - $\phi_M^n$')
#ax[0].plot(frecs_M,X,'o-',label='Fases')
ax[0].plot(frecs_M,X_equiv,'s-',label='Fases equiv')
ax[0].plot(frecs_M,np.arctan(Y),'o-',label='arctan tan')
ax[0].axhline(np.pi/2,0,1,ls='--',c='k',label='$\pi/2$')

ax[1].set_title('tan(n*$\phi_H$ - $\phi_M^n$)')
ax[1].plot(frecs_M,Y,'o-',label='tan(fases)')
ax[1].plot(frecs_M,np.tan(X_equiv),'.-',label='tan(fases equiv)')


for a in ax:
    a.legend()
    a.grid()
ax[1].set_xticks(frecs_M)    
ax[1].set_xlabel('frec')
plt.savefig('fase_tan(fase)_simulado.png')
    
#%% FFT desde 0

# frecs_H=rfftfreq(len(H),d=2e-8)
# frecs_H=frecs_H[np.nonzero(frecs_H<=15*10000)]
# g_H = fft(H)
# g_H = np.resize(g_H,len(frecs_H))
# mag_H=abs(g_H)
# fase_H=np.angle(g_H)

# frecs_M=rfftfreq(len(M),d=2e-8)
# frecs_M=frecs_M[np.nonzero(frecs_M<=15*10000)]
# g_M = fft(M)
# g_M = np.resize(g_M,len(frecs_M))
# mag_M=abs(g_M)
# fase_M=np.angle(g_M)

# picos,_=find_peaks(mag_M,threshold=max(mag_M)*0.01)

# fig,(a,b)=plt.subplots(nrows=2,sharex=True)
# # a.plot(frecs_H,mag_H,'.-')
# a.plot(frecs_M,mag_M,'.-')
# a.scatter(frecs_M[picos],mag_M[picos])

# b.scatter(frecs_H[2],fase_H[2],c='r')
# b.scatter(frecs_M[picos],fase_M[picos]%np.pi)
# for ax in [a,b]:
#     ax.grid()
# #b.set_xticks(frecs_H)    

# plt.show()

# fig,ax=plt.subplots(nrows=2,constrained_layout=True,sharex=True)
# ax[0].stem(frecs_aux,mag_M_Pedro/max(mag_M_Pedro),markerfmt ='o',basefmt=' ',label='M')
# ax[0].stem(frecs_aux,mag_H_Pedro/max(mag_H_Pedro),markerfmt ='r.',basefmt=' ',label='H')
# ax[1].scatter(frecs_aux,fases_M_Pedro,label='M')
# # ax[1].scatter(frecs_aux,fases_H_Pedro,label='H')
# ax[1].set_ylim(0,np.pi)
# ax[1].set_xlim(0,1e5)
# # ax[1].plot(frecs_M,y_2,'o-',label='tan(n*Fase_H - phi0_M)')
# # ax.plot(frec,Fase_m2,'.-',label='Phi ')
# # ax.plot(frec,np.tan(Fase_m2),'.-',label='Tan(phi)')
# for a in ax:
#     a.legend()
#     a.grid()
# ax[1].set_xticks(frecs_aux)    
# ax[1].set_xlabel('frec')
# plt.show()

# y_1 = []
# for n in range(len(frecs_aux)):
#     y_1.append(((2*n+1)*fases_H_Pedro[0]-fases_M_Pedro[n]))
#     print(2*n+1,f'frec: {frecs_aux[n]:8.0f} Hz -',f'tau: {np.tan(((2*n+1)*fases_H_Pedro[0]-fases_M_Pedro[n]))/(2*np.pi*frecs_aux[0]):.2e}')
# y_1=np.array(y_1)
# y_2=np.tan(y_1)

#%
# fig,ax=plt.subplots(nrows=2,constrained_layout=True,sharex=True)
# #ax[0].set_title('n*Fase_H - phi0_M')
# ax[0].plot(frecs_aux,y_1,'o-',label='n*Fase_C - Fase_M')
# #ax[0].plot(frecs_M,y_3,label='2*pi*tau*f')
# a#x[0].set_ylim(0,np.pi/2 + 0.013)
# #ax[1].set_title('tan(n*Fase_H - phi0_M)')
# ax[1].plot(frecs_aux,y_2,'o-',label='tan(n*Fase_H - phi0_M)')
# # ax.plot(frec,Fase_m2,'.-',label='Phi ')
# # ax.plot(frec,np.tan(Fase_m2),'.-',label='Tan(phi)')
# for a in ax:
#     a.legend()
#     a.grid()
# ax[1].set_xticks(frecs_aux)    
# ax[1].set_xlabel('frec')


# %% 25 Mar 25 - Voy con files reales

# Leer el archivo y cargar los datos en un array de NumPy
def cargar_obtener_fases(archivo):
    data = np.loadtxt(archivo,skiprows=9, usecols=(0,3,4))

    # Separar las columnas en tres arrays diferentes
    t = data[:, 0]  # Primera columna
    H = data[:, 1]  # Segunda columna
    M = data[:, 2]  # Tercera columna

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(t,H/max(H))
    ax.plot(t,M/max(M))
    ax.set_xlabel('t')
    ax.set_title('H',loc='left')
    ax.set_title('M',loc='left')
    ax.grid()
    plt.xlim(0,t[-1])
    #plt.savefig('señal_cruda.png')
    plt.show()
    plt.plot(H,M)
    # t=t-t[0]

    offset , amp, frec , fase = ajusta_seno(t,H)
    print(offset , amp, frec , fase)

    y=sinusoide(t,offset , amp, frec , fase)
    print('FFT del script mio')
    _, _, muestra_rec_impar,delta_phi_0,f_0,amp_0,fase_0, espectro_f_amp_fase_m,espectro_ref = fourier_señales_5(t,M,H,
                                                                                                            delta_t=2e-8,polaridad=1,
                                                                                                            filtro=0.05,frec_limite=13*frec,
                                                                                                            name='test')
    #plt.close('all')
    # fig_fourier.savefig(os.path.join(os.getcwd(),'simulado_Espectro.png'),dpi=200,facecolor='w')
    # fig2_fourier.savefig(os.path.join(os.getcwd(),'simulado_Rec_impar.png'),dpi=200,facecolor='w')

    #FASE
    frecs_M=espectro_f_amp_fase_m[0]
    frecs_H=espectro_ref[0]

    Fase_H=espectro_ref[2]
    Fase_M= espectro_f_amp_fase_m[2]

    print('''Espectro de la señal de Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    print(f"{frecs_H:.8e} {espectro_ref[1]:.8e} {Fase_H:.8e}")

    print('''\nEspectro de la señal de muestra:\nFrecuencia_Hz|Intensidad rel|Dphi''')
    for i in range(len(frecs_M)):
        print(f'{frecs_M[i]:.8e} {espectro_f_amp_fase_m[1][i]:.8e} {Fase_M[i]:.8e}')
    
    tau_0=np.tan(Fase_H-Fase_M[0])/(2*np.pi*frecs_M[0])
    print('\ntau (1er armonico)= tan(Fase_H - phi0_M)/(2*pi*f0) = ',f'{tau_0:.2e} s\n')
    return frecs_M,frecs_H,Fase_H,Fase_M

#%%
#La operacion de tg y arcotangente te lo manda al rango necesario
def reducir_a_rango_principal_numpy(x):
    '''np.fmod(x + np.pi / 2, np.pi): Calcula el módulo de x+π/2 con respecto a π. 
    Esto asegura que el resultado esté en el intervalo [0,π).'''
    x_mod = np.fmod(x + np.pi / 2, np.pi) # Paso 1: Sumar pi/2 y calcular el módulo pi
    
    x_equiv = x_mod - np.pi / 2 # Paso 2: Restar pi/2 para llevar al rango (-pi/2, pi/2)
      
    return x_equiv

#%% files 
path_NF_citrato=os.path.join('real_files','265kHz_150dA_100Mss_bobN1NFcitrato_ciclo_promedio_H_M.txt')
path_NF=os.path.join('real_files','265kHz_150dA_100Mss_bobN1NF_ciclo_promedio_H_M.txt')
path_NE=os.path.join('real_files','265kHz_150dA_100Mss_bobN1NE5X0_ciclo_promedio_H_M.txt')
#%%
path_NF=os.path.join('real_files','265kHz_150dA_100Mss_bobN1NF0019.txt') #NF241121 Citrato
# %% Laburo con los espectros obtenidos del procesador
#def cargar_obtener_fase(path):
path_espectro_NF=os.path.join('real_files','265kHz_150dA_100Mss_bobN1NF0019_Espectro.txt')
data_H = np.loadtxt(path_espectro_NF,skiprows=4,max_rows=1, usecols=(0,1,2))
data_M = np.loadtxt(path_espectro_NF,skiprows=7, usecols=(0,1,2))

#%%
# Separar las columnas en tres arrays diferentes
f_H,amp_H,fase_H = data_H[0],data_H[1],data_H[2]
f_M = data_M[:, 0]  
amp_M = data_M[:, 1]  
fase_M = data_M[:, 2]  

fase_M_new=np.arctan(np.tan(reducir_a_rango_principal_numpy(fase_M)))

fig,(ax,ax2)=plt.subplots(nrows=2,constrained_layout=True,figsize=(8,6),sharex=True)
ax.plot(f_M,amp_M,'o-')
ax.set_xlabel('f')
ax.grid()
ax2.plot(f_M,fase_M,'o-')
ax2.plot(f_M,fase_M_new,'o-')

ax2.set_xlabel('H')
ax2.set_ylabel('M')
ax2.grid()
plt.show()

fig,ax=plt.subplots(nrows=2,constrained_layout=True,sharex=True)
ax[0].set_title('n*$\phi_H$ - $\phi_M^n$')
#ax[0].plot(frecs_M,X,'o-',label='Fases')
ax[0].plot(f_M,fase_M_new,'s-',label='Fases equiv')
#ax[0].plot(f_M,np.arctan(Y),'o-',label='arctan tan')
ax[0].axhline(np.pi/2,0,1,ls='--',c='k',label='$\pi/2$')

ax[1].set_title('tan(n*$\phi_H$ - $\phi_M^n$)')
# ax[1].plot(f_M,Y,'o-',label='tan(fases)')
# ax[1].plot(f_M,np.tan(X_equiv),'.-',label='tan(fases equiv)')


for a in ax:
    a.legend()
    a.grid()
#ax[1].set_xticks(frecs_M)    
ax[1].set_xlabel('frec')
plt.savefig('fase_tan(fase)_simulado.png')

offset , amp, frec , fase = ajusta_seno(t,H)
print(offset , amp, frec , fase)

frecs_H=rfftfreq(len(H),d=2e-8)
frecs_H=frecs_H[np.nonzero(frecs_H<=15*10000)]
g_H = fft(H)
g_H = np.resize(g_H,len(frecs_H))
mag_H=abs(g_H)
fase_H=np.angle(g_H)

frecs_M=rfftfreq(len(M),d=2e-8)
frecs_M=frecs_M[np.nonzero(frecs_M<=15*10000)]
g_M = fft(M)
g_M = np.resize(g_M,len(frecs_M))
mag_M=abs(g_M)
fase_M=np.angle(g_M)

picos,_=find_peaks(mag_M,threshold=max(mag_M)*0.01)

fig,(a,b)=plt.subplots(nrows=2,sharex=True)
a.plot(frecs_H,mag_H,'.-')
a.plot(frecs_M,mag_M,'.-')
a.scatter(frecs_M[picos],mag_M[picos])

# b.scatter(frecs_H[2],fase_H[2],c='r')
# b.scatter(frecs_M[picos],fase_M[picos]%np.pi)
for ax in [a,b]:
    ax.grid()
#b.set_xticks(frecs_H)    

plt.show()

# _, _, muestra_rec_impar,delta_phi_0,f_0,amp_0,fase_0, espectro_f_amp_fase_m,espectro_ref = fourier_señales_5(t,M,H,
#                                                                                                         delta_t=2e-8,polaridad=1,
#                                                                                                         filtro=0.05,frec_limite=13*frec,
#                                                                                                         name='test')
# #FASE
# frecs_M=espectro_f_amp_fase_m[0]
# frecs_H=espectro_ref[0]

# Fase_H=espectro_ref[2]
# Fase_M= espectro_f_amp_fase_m[2]

# print('''Espectro de la señal de Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
# print(f"{frecs_H:.8e} {espectro_ref[1]:.8e} {Fase_H:.8e}")

# print('''\nEspectro de la señal de muestra:\nFrecuencia_Hz|Intensidad rel|Dphi''')
# for i in range(len(frecs_M)):
#     print(f'{frecs_M[i]:.8e} {espectro_f_amp_fase_m[1][i]:.8e} {Fase_M[i]:.8e}')

# tau_0=np.tan(Fase_H-Fase_M[0])/(2*np.pi*frecs_M[0])
# print('\ntau (1er armonico)= tan(Fase_H - phi0_M)/(2*pi*f0) = ',f'{tau_0:.2e} s\n')
