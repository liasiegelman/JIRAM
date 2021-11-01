import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys, glob, os
import cmocean
from matplotlib import colors
import sys, glob
import scipy.io as io
from scipy import ndimage, misc, signal, stats
from matplotlib.colors import LogNorm
import getpass
user = getpass.getuser()
# if user == 'lsiegelma' :
#     sys.path.append('/Users/lsiegelma/pds-tools/')
sys.path.append('/Users/'+user+'/Dropbox/Jovian_dynamics/pds-tools/')
from vicar import VicarImage
sys.path.append('/Users/'+user+'/Dropbox/Jovian_dynamics/programs/')
import Powerspec_a as ps
import jupiter_functions_a as jf
import twodspec  as spec
from scipy.stats import skew

import math
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Box2DKernel
from datetime import date
## make repertory for today's plots
today = date.today()
d1 = today.strftime("%Y%m%d")
dir_out = '/Users/'+user+'/Dropbox/Jovian_dynamics/plot/'+d1+'/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
# sys.path.append('/Users/lsiegelma/Documents/Kerguelen_analysis/programs_hector/')
# import TwoD_wavenumber_spec as spec



os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin' # this is to be able to use latex - need to specify this better with PATH command at some point
import subprocess
subprocess.check_call(["latex"])


## defining parameters for Jupiter atmosphere ##
g = 29.8
f0 = 3.5e-4
N0 = 4e-3
alpha = 9000*N0**2/g ## as it is it is now equivalent as choosing H (and to retrieve PE we do H*N0*tau) #0.005 #0.007#0.0055 ### 0.024 when optimized on eke for 1000 km , 0.019 when optimized on psi  for 1000 km0.019  ## 0.0XX for 2OOO km and 0.019 for 1000 km
# alpha = 9500*N0**2/g #H0 = 9500 for n0204
## resolution of the data, here 10km/pixel
dx = 1e4
dy = 1e4

## upload infrared image and make it doubly periodic ##
# dir = '/Users/lsiegelma/Documents/Jovian_dynamics/data/feature_tracking/mosaic/'
dir = '/Users/'+user+'/Dropbox/Jovian_dynamics/data/feature_tracking/mosaic/'
# u_load = np.load(dir+'n0103/u_n0103_15km.npy') # 15km/pixel data
# v_load = np.load(dir+'n0103/v_n0103_15km.npy') # 15km/pixel data
# im = np.load(dir+'im_n02_15km.npy') # 15km/pixel data

# u_load = np.load(dir+'n0103/u_n0103.npy') # 10km/pixel data we used these data to plot the final KE transfer and spectra
# v_load = np.load(dir+'n0103/v_n0103.npy') # 10km/pixel data
# im = np.load(dir+'n0103/im_n02.npy') # 10km/pixel data

# u_load = np.load(dir+'n0204/u_n0204.npy') # 10km/pixel data for n0204
# v_load = np.load(dir+'n0204/v_n0204.npy') # 10km/pixel data
# im = np.load(dir+'n0204/im_n03.npy') # 10km/pixel data

u_load = np.load(dir+'n0103/u_n0103_nov9_10km.npy') # 10km/pixel data nov 9
v_load = np.load(dir+'n0103/v_n0103_nov9_10km.npy') # 10km/pixel data nov 9
im = np.load(dir+'n0103/im_n02_oct30_10km.npy') # 10km/pixel data oct 30
# im[im==0]=sys.float_info.epsilon

## get rids of the zeros - otherwise causes problem in the spectra
list = np.argwhere(im==0)
for i in range(np.argwhere(im==0).shape[0]):
    im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
np.sum(im==0)

I_0 = np.max(im)
Tau = np.log10(I_0/im)

## make the data doubly periodic ##
tau_per = jf.doubly_per(Tau)
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
Tau = jf.detrend_data(Tau, detrend='Both')
u_l = jf.detrend_data(u_load, detrend='Both')
v_l = jf.detrend_data(v_load, detrend='Both')

tau = jf.detrend_data(tau_per, detrend='Both')
u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N



## Figure 2 - "Dynamical context"
n = 1
wc = 250 # 200 for n0204

## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
## retrieve psi and co from u and v ##
PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

lim = 1 ## limit for imshow
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
plt.imshow(Tau, origin='lower',cmap ='bone', vmin = -0.7, vmax = 1.)
plt.colorbar(shrink = 0.4, label = r'$\tau$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta, k < 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)
V = VORT_b[im.shape[0]:,:im.shape[1],1].real
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
plt.imshow(V_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.colorbar(shrink = 0.4, label = r'$\zeta$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}, k > 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)
TK = VORT_tau[im.shape[0]:,:im.shape[1],1].real
spec_thp, TK_filt = jf.cut_hp(dx,dy,TK,6e-4*1e-3)
plt.imshow(TK_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -3.5, vmax = 3.5)
plt.colorbar(shrink = 0.4, label = r'$\zeta_{\tau}$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()
fig.savefig(dir_out+'Fig2_lim_'+str(lim)+'_n0103.pdf', bbox_inches='tight')
plt.close(fig)

## Figue 3 - "Spectral characteristics"
u_load = np.load(dir+'n0103/u_n0103.npy') # 10km/pixel data we used these data to plot the final KE transfer and spectra
v_load = np.load(dir+'n0103/v_n0103.npy') # 10km/pixel data
im = np.load(dir+'n0103/im_n02.npy') # 10km/pixel data

## get rids of the zeros - otherwise causes problem in the spectra
list = np.argwhere(im==0)
for i in range(np.argwhere(im==0).shape[0]):
    im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
np.sum(im==0)

I_0 = np.max(im)
Tau = np.log10(I_0/im)

## make the data doubly periodic ##
tau_per = jf.doubly_per(Tau)
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
Tau = jf.detrend_data(Tau, detrend='Both')
u_l = jf.detrend_data(u_load, detrend='Both')
v_l = jf.detrend_data(v_load, detrend='Both')

tau = jf.detrend_data(tau_per, detrend='Both')
u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N

n = 1
wc = 250 # 200 for n0204

## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
## retrieve psi and co from u and v ##
PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
kr , E_div = jf.wv_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)

spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
KE = 0.5*(E_u+E_v)

kr , E_tau = jf.wv_spec(Tau,dx,dy,detrend=None)

T = tau
Nj,Ni = T.shape
_,wavnum2D,kx,ky = jf.wvnumb_vector(dx,dy,Ni,Nj)
spec_tk = np.fft.fft2(T)*wavnum2D*2*np.pi
TK = np.fft.ifft2(spec_tk).real

kr , E_tk = jf.wv_spec(TK[im.shape[0]:,:im.shape[1]],dx,dy,detrend=None)
kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)

### running mean
VAR = Eke_prod*kr
N= 17 # moving average window N = 10 corresponds to 1 km
Q_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    Q_e = c.squeeze()
Q_e[0]=Eke_prod[0]*kr[0]
Q_e[-1]=Eke_prod[-1]*kr[-1]

VAR = E_div/E_vort
N=5 # moving average window N = 10 corresponds to 1 km
R_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    R_e = c.squeeze()
R_e[0]=E_div[0]/E_vort[0]
R_e[-1]=E_div[-1]/E_vort[-1]

## Modified figure spectra according to the reviewer 3 recommendation
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(12,24))

ax1 = fig.add_subplot(311)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-3/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{APE}$') #label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
ax1.loglog(kr*1e3,KE*1e-3/np.pi, lw=3, label = r'$\mathrm{KE}$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1],  labels[0]]
ax1.legend(handles,labels,frameon=False,loc=3)
ks = np.array([2.5e-4,5.2e-4])
es = 6e-6*ks**-3
ax1.loglog(ks,es,'k--')
ax1.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
es = 1.1e0*ks**-(4/3)
ax1.loglog(ks,es,'k--')
ax1.text(5e-4,2.2e6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.text(2e-3,1.3e4,r'$k^{-4/3}$')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax1.set_ylim((1e1,1.3e6))
ax1.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(312)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-3/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
ax1.loglog(kr*1e3,E_div*1e-3/np.pi, lw=3, label = r'$\chi$', c='tab:green')
ax1.loglog(kr*1e3,E_vort*1e-3/np.pi, lw=3, label = r'$\zeta$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
ax1.legend(handles,labels,frameon=False)
ks = np.array([2.5e-4,5.2e-4])
es = 9e-11*ks**(-1.1)
ax1.loglog(ks,es,'k--')
ax1.text(4e-4,7e-7,r'$k^{-1}$',fontsize=18)
ks = np.array([7e-4,5e-3])
es = 3.3e-5*ks**(2/3)
ax1.loglog(ks,es,'k--')
ax1.text(1.4e-3,6.5e-7,r'$k^{2/3}$',fontsize=18)
ax1.set_ylim((5e-9,1.5e-6))
ax1.text(5e-4,1.95e-6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(313)
ax2 = ax1.twiny()
ax1.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.axhline(y=1,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')
plt.tight_layout()

fig.savefig(dir_out+'Figure3_n0103_n1_cutoff_250_H0_9000.pdf', bbox_inches='tight')


## figure 4 KE and ens transfer
## enstrophy flux ##
kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')
# kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend='Both') # for n0204


plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(121)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(122)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

plt.tight_layout()

fig.savefig(dir_out+'Figure4_n0103_n1_cutoff_250_H0_9000.pdf', bbox_inches='tight')
# fig.savefig(dir_out+'Figure4_n0204_n1_cutoff_200_H0_9500.png', bbox_inches='tight')


## Extended Data Figures ##

## Extended Data Figure 1 ##
u_load = np.load(dir+'n0103/u_n0103_nov9_10km.npy') # 10km/pixel data nov 9
v_load = np.load(dir+'n0103/v_n0103_nov9_10km.npy') # 10km/pixel data nov 9
im = np.load(dir+'n0103/im_n02_oct30_10km.npy') # 10km/pixel data oct 30
# im[im==0]=sys.float_info.epsilon

## get rids of the zeros - otherwise causes problem in the spectra
list = np.argwhere(im==0)
for i in range(np.argwhere(im==0).shape[0]):
    im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
np.sum(im==0)

I_0 = np.max(im)
Tau = np.log10(I_0/im)

## make the data doubly periodic ##
tau_per = jf.doubly_per(Tau)
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
tau = jf.detrend_data(tau_per, detrend='Both')
u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
Int = 10
lim = 80
plt.title(r'$\mathrm{Wind \: velocity}$')
n = 2
wc = 500
## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
U = filt_u[im.shape[0]:,:im.shape[1]]
V = filt_v[im.shape[0]:,:im.shape[1]]
M = np.sqrt(U**2+V**2)
M[M>lim]=lim
Q = plt.quiver(U[::Int,::Int], V[::Int,::Int],M[::Int,::Int],scale = 700, units = 'width', cmap = cmocean.cm.speed)
plt.quiverkey(Q, 0.24, 0.97, 50, r'50 $\mathrm{m.s}^{-1}$',labelpos ='E',coordinates ='figure')
plt.colorbar(shrink = 0.4,label = r'$\mathrm{Wind \: velocity} \; [\mathrm{m.s}^{-1}]$')
plt.xlim((0,U.shape[1]/Int))
plt.ylim((0,U.shape[0]/Int))
plt.xlabel(r'$\mathrm{x} \: [100  \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [100  \mathrm{km}]$')


plt.subplot(132)
n = 1
wc = 250
lim=2
## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)


PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)
V = VORT_b[im.shape[0]:,:im.shape[1],1].real
plt.imshow(V*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.colorbar(shrink = 0.4, label = r'$\zeta$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
D = DIV_b[im.shape[0]:,:im.shape[1],1].real

plt.imshow(D*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.colorbar(shrink = 0.4, label = r'$\chi$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()

fig.savefig(dir_out+'Extended_Data_Fig_1_n0103.jpg', bbox_inches='tight')
plt.close(fig)


## Extended Data Figure 2 - Rossby distribution ##
u_load = np.load(dir+'n0103/u_n0103_nov9_10km.npy') # 10km/pixel data nov 9
v_load = np.load(dir+'n0103/v_n0103_nov9_10km.npy') # 10km/pixel data nov 9
im = np.load(dir+'n0103/im_n02_oct30_10km.npy') # 10km/pixel data oct 30
# im[im==0]=sys.float_info.epsilon

## get rids of the zeros - otherwise causes problem in the spectra
list = np.argwhere(im==0)
for i in range(np.argwhere(im==0).shape[0]):
    im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
np.sum(im==0)

I_0 = np.max(im)
Tau = np.log10(I_0/im)

## make the data doubly periodic ##
tau_per = jf.doubly_per(Tau)
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
tau = jf.detrend_data(tau_per, detrend='Both')
u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N

V = VORT_b[u_load.shape[0]:,:u_load.shape[1],1].real

plt.rcParams['font.size'] = '18'
fig = plt.figure(figsize=(20,5))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# spec_thp, V_filt = jf.cut_lp(dx,dy,V,(1/250)*1e-3)
S=skew(V.flatten())
# S=skew(V_filt.flatten())
n, bins, patches = plt.hist(V.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
plt.title('$\mathrm{Full} \: \mathrm{field}$')
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,3e5))

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k < 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
# plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-0.6,0.6))
plt.ylim((1e2,3e5))
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
# plt.tight_layout()

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
# spec_thp, V_filt = jf.cut_bp(dx,dy,V,(1/250)*1e-3,(1/2500)*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k > 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,3e5))

plt.tight_layout()
fig.savefig(dir_out+'rossby_distribution.pdf', bbox_inches='tight')



## Extended Data Figure 3 - "Band pass filtered tau, zeta_tau, zeta, chi"
V = VORT_b[im.shape[0]:,:im.shape[1],1].real
D = DIV_b[im.shape[0]:,:im.shape[1],1].real
TK = VORT_tau[im.shape[0]:,:im.shape[1],1].real
TAU = tau[im.shape[0]:,:im.shape[1]]


wl_low = 250
wl_high = (6e-4)

spec_thp, V_filt = jf.cut_bp(dx,dy,V, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, D_filt = jf.cut_bp(dx,dy,D, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, TK_filt = jf.cut_bp(dx,dy,TK, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, TauK_filt = jf.cut_bp(dx,dy,TAU, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)


coord = [[600,789], [600,200], [1100,200], [1100,789]] ## coordinates of the polar vortex
coord.append(coord[0]) #repeat the first point to create a 'closed loop'
xs, ys = zip(*coord)


coord_f1 = [[0,260], [0,0], [300,0], [300,260]] ## coordinates of the filament in the lowerleft coner
coord_f1.append(coord_f1[0]) #repeat the first point to create a 'closed loop'
xs_f1, ys_f1 = zip(*coord_f1)


coord_f3 = [[1500,100], [2000,100], [2000,600], [1500,600]] ## coordinates of a region we dont use (I dont remember which region)
coord_f3.append(coord_f3[0]) #repeat the first point to create a 'closed loop'
xs_f3, ys_f3 = zip(*coord_f3)


coord_f2 = [[1280,170], [1500,650], [1110,650], [900,170]] ## coordinates of the famous streamer
coord_f2.append(coord_f2[0]) #repeat the first point to create a 'closed loop'
xs_f2, ys_f2 = zip(*coord_f2)


coord_f4 = [[1,2], [2078,2], [2078,789], [1,789]] ## coordinates of the full field
coord_f4.append(coord_f4[0]) #repeat the first point to create a 'closed loop'
xs_f4, ys_f4 = zip(*coord_f4)


## filtered fields in the physical space
lim = 2 ## limit for imshow
lw = 5 ## contour width for the subdomains
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(8*w,1.5*h))

plt.subplot(141)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
plt.imshow(TauK_filt,origin='lower', vmin = -0.55, vmax = 0.55, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4,label =r'$\tau$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(142)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}$', fontsize = 34)
plt.imshow(TK_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$\zeta_{\tau}$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(143)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)

plt.imshow(V_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$\zeta$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(144)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
plt.imshow(D_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='green')
plt.colorbar(shrink = 0.4, label = r'$\chi$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.tight_layout()
fig.savefig(dir_out+'filtered_fields_lim_'+str(lim)+'_4fields.pdf', bbox_inches='tight')
plt.close(fig)



## Extended Data Figure 4 --6 , scatter plots
var = V_filt
M_vort = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TK_filt[0:300,0:260]*1e4
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[600:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TK_filt[600:1100,200:-1]*1e4
    if j==2:
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]*1e4
    if j==3:
        Y = var*1e4 ## full field
        X = TK_filt*1e4

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    M_vort[:,0,j]=val_m
    M_vort[:,1,j]=val_my
    M_vort[:,2,j]=var_my

var = D_filt
M_div = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TK_filt[0:300,0:260]*1e4
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[500:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TK_filt[500:1100,200:-1]*1e4
    if j==2:
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]*1e4
    if j==3:
        Y = var*1e4 ## full field
        X = TK_filt*1e4

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    M_div[:,0,j]=val_m
    M_div[:,1,j]=val_my
    M_div[:,2,j]=var_my


## scatter plots between chi and tau - done during the revision stage
var = D_filt
T_div = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TauK_filt[0:300,0:260]
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[500:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TauK_filt[500:1100,200:-1]
    if j==2:
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TauK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]
    if j==3:
        Y = var*1e4 ## full field
        X = TauK_filt

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    T_div[:,0,j]=val_m
    T_div[:,1,j]=val_my
    T_div[:,2,j]=var_my



### plot the 4 squares for zeta and zeta_tau##
fac = 2 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_vort[:,0,1],M_vort[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],M_vort[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], M_vort[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,1][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,1][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,0][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,0][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-1.2,1.45))
plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_vort.pdf', bbox_inches='tight')



### plot the 4 squares for div and zeta_tau##
fac = 2 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_div[:,0,1],M_div[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],M_div[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], M_div[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)[1:]
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.hstack((-1.7,np.linspace(-2,2,11)[2:]))
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim((-1.2,1.45))
# plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_div.pdf', bbox_inches='tight')


### plot the 4 squares for div and tau##
fac = 1 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_div[:,0,1],M_div[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],T_div[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], T_div[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [ \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.4,0.65,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-0.4,0.6))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.98,0.6,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-2,2.5))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.7,0.7,11)[1:]
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.hstack((np.linspace(-1,1.3,11)[0:-4],0.61))
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-0.7,1.5))
# plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_div_tau_bis.pdf', bbox_inches='tight')


## Extended Data Figure 9 - sensitivity analysis


## add noise 20 times (or more with the ite variable)
u_loadd = np.load(dir+'n0103/u_n0103.npy') # 10km/pixel data we used these data to plot the final KE transfer and spectra
v_loadd = np.load(dir+'n0103/v_n0103.npy') # 10km/pixel data
im = np.load(dir+'n0103/im_n02.npy') # 10km/pixel data

ite = 20
KE_MAT = np.zeros((369,ite)) # KE.shape[0]=369
Ediv_MAT = np.zeros((369,ite))
Evort_MAT = np.zeros((369,ite))
Re_MAT = np.zeros((369,ite))
Qe_MAT = np.zeros((369,ite))
Tpi_MAT = np.zeros((369,ite))

## add noise to u and v
mean_noise = 0
mu_noise = (2.1)**2 ## the standard deviation is 2.1 m/s from Tricia's analysis

for ji in range(ite): ## takes ~14 minutes to complete

    if ji == 0:
         u_load = u_loadd
         v_load = v_loadd

    else :
        noise = np.random.normal(mean_noise, np.sqrt(mu_noise), u_load.shape)

        u_load = u_loadd + noise
        v_load = v_loadd + noise

    ## get rids of the zeros - otherwise causes problem in the spectra
    list = np.argwhere(im==0)
    for i in range(np.argwhere(im==0).shape[0]):
        im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
    np.sum(im==0)

    I_0 = np.max(im)
    Tau = np.log10(I_0/im)

    ## make the data doubly periodic ##
    tau_per = jf.doubly_per(Tau)
    u_per = jf.doubly_per(u_load)
    v_per = jf.doubly_per(v_load)

    ## detrend the data ##
    Tau = jf.detrend_data(Tau, detrend='Both')
    u_l = jf.detrend_data(u_load, detrend='Both')
    v_l = jf.detrend_data(v_load, detrend='Both')

    tau = jf.detrend_data(tau_per, detrend='Both')
    u = jf.detrend_data(u_per, detrend='Both')
    v = jf.detrend_data(v_per, detrend='Both')

    PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N

    ## Figure 2
    n = 1
    wc = 250 # 200 for n0204

    ## apply a butterworth low pass to u and v ##
    spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
    spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
    ## retrieve psi and co from u and v ##
    PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

    kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
    kr , E_div = jf.wv_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
    # for n0204
    kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)
    kr , E_div = jf.wv_spec(DIV_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)

    spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
    spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

    # for n0204
    # u_l = u_l[1:,1:]
    # v_l = v_l[1:,1:]
    # spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
    # spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

    kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
    kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
    KE = 0.5*(E_u+E_v)

    kr , E_tau = jf.wv_spec(Tau,dx,dy,detrend=None)
    # kr , E_tau = jf.wv_spec(Tau[1:,1:],dx,dy,detrend=None) # for n0204


    T = tau
    Nj,Ni = T.shape
    _,wavnum2D,kx,ky = jf.wvnumb_vector(dx,dy,Ni,Nj)
    spec_tk = np.fft.fft2(T)*wavnum2D*2*np.pi
    TK = np.fft.ifft2(spec_tk).real

    kr , E_tk = jf.wv_spec(TK[im.shape[0]:,:im.shape[1]],dx,dy,detrend=None)
    # kr , E_tk = jf.wv_spec(TK[im.shape[0]+1:,1:im.shape[1]],dx,dy,detrend=None) # for n0204


    kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)
    # kr,tpi,Eke_prod = jf.KE_flux_doubly_per(filt_u,filt_v,dx,dy,detrend=None) # at some point with Patrice we explored in double periodic and setting some part to 0 but I am not sure I did it properly - should probably look more into the symmetry of the streamfunction to see if I did u = -u correctly or not .... Maybe something for follow up paper...


    ### running mean
    VAR = Eke_prod*kr
    N= 17 # moving average window N = 10 corresponds to 1 km
    Q_e = np.empty((VAR.shape[0]))*np.nan
    for j in range(len(VAR)):
        bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
        c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
        Q_e = c.squeeze()
    Q_e[0]=Eke_prod[0]*kr[0]
    Q_e[-1]=Eke_prod[-1]*kr[-1]

    VAR = E_div/E_vort
    N=5 # moving average window N = 10 corresponds to 1 km
    R_e = np.empty((VAR.shape[0]))*np.nan
    for j in range(len(VAR)):
        bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
        c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
        R_e = c.squeeze()
    R_e[0]=E_div[0]/E_vort[0]
    R_e[-1]=E_div[-1]/E_vort[-1]

    kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')

    KE_MAT[:,ji] = KE
    Ediv_MAT[:,ji] = E_div
    Evort_MAT[:,ji] = E_vort
    Re_MAT[:,ji] = R_e
    Qe_MAT[:,ji] = Q_e
    Tpi_MAT[:,ji] = tpi_vort

    print(str(ji+1)+"/"+str(ite)+' is done')

# construc enveloppe
env_up_KE = np.zeros(KE_MAT.shape[0])
env_up_Ediv = np.zeros(Ediv_MAT.shape[0])
env_up_Evort = np.zeros(Evort_MAT.shape[0])
env_up_Re = np.zeros(Re_MAT.shape[0])
env_up_Qe = np.zeros(Qe_MAT.shape[0])
env_up_Tpi = np.zeros(Tpi_MAT.shape[0])

env_down_KE = np.zeros(KE_MAT.shape[0])
env_down_Ediv = np.zeros(Ediv_MAT.shape[0])
env_down_Evort = np.zeros(Evort_MAT.shape[0])
env_down_Re = np.zeros(Re_MAT.shape[0])
env_down_Qe = np.zeros(Qe_MAT.shape[0])
env_down_Tpi = np.zeros(Tpi_MAT.shape[0])

for i in range(KE_MAT.shape[0]):
    env_up_KE[i] = np.max(KE_MAT[i,:])
    env_up_Ediv[i] = np.max(Ediv_MAT[i,:])
    env_up_Evort[i] = np.max(Evort_MAT[i,:])
    env_up_Re[i] = np.max(Re_MAT[i,:])
    env_up_Qe[i] = np.max(Qe_MAT[i,:])
    env_up_Tpi[i] = np.max(Tpi_MAT[i,:])

    env_down_KE[i] = np.min(KE_MAT[i,:])
    env_down_Ediv[i] = np.min(Ediv_MAT[i,:])
    env_down_Evort[i] = np.min(Evort_MAT[i,:])
    env_down_Re[i] = np.min(Re_MAT[i,:])
    env_down_Qe[i] = np.min(Qe_MAT[i,:])
    env_down_Tpi[i] = np.min(Tpi_MAT[i,:])


## explore the sensitivity to the butterworh filter
u_load = np.load(dir+'n0103/u_n0103.npy') # 10km/pixel data we used these data to plot the final KE transfer and spectra
v_load = np.load(dir+'n0103/v_n0103.npy') # 10km/pixel data

## make the data doubly periodic ##
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
u_l = jf.detrend_data(u_load, detrend='Both')
v_l = jf.detrend_data(v_load, detrend='Both')

u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

n = 1
wc_list= [50,150,250,500]
MAT_tpi = np.empty((len(wc_list)*len(n_list), 369))
MAT_qe = np.empty((len(wc_list)*len(n_list), 369))
I=0
for wc in wc_list:
    print(wc)
    print("I = "+str(I))
    ## apply a butterworth low pass to u and v ##
    spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
    spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
    ## retrieve psi and co from u and v ##
    PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

    kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
    kr , E_div = jf.wv_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)

    spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
    spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

    kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
    kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
    KE = 0.5*(E_u+E_v)

    kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)
    kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')

    ### running mean
    VAR = Eke_prod*kr
    N = 17 # moving average window N = 10 corresponds to 1 km
    Q_e = np.empty((VAR.shape[0]))*np.nan
    for j in range(len(VAR)):
        bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
        c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
        Q_e = c.squeeze()
    Q_e[0]=Eke_prod[0]*kr[0]
    Q_e[-1]=Eke_prod[-1]*kr[-1]

    MAT_qe[I,:] =  Q_e
    MAT_tpi[I,:] =  tpi_vort
    I=I+1

# normalize matrix per respect to curve in main manuscript
c_50 = MAT_qe[0,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[0,:]))
c_100 = MAT_qe[1,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[1,:]))
c_500 = MAT_qe[-1,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[-1,:]))

tpi_50 = MAT_tpi[0,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[0,:]))
tpi_100 = MAT_tpi[1,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[1,:]))
tpi_500 = MAT_tpi[-1,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[-1,:]))



## Modified figure spectra according to the reviewer 3 recommendation
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,32))

ax1 = fig.add_subplot(421)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.fill_between(kr*1e3,env_down_KE*1e-3/np.pi, env_up_KE*1e-3/np.pi, color='grey', alpha = 0.5, label=r'$\mathrm{KE}$')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(frameon=False)
ks = np.array([2.5e-4,5.2e-4])
es = 6e-6*ks**-3
ax1.loglog(ks,es,'k--')
ax1.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
es = 1.1e0*ks**-(4/3)
ax1.loglog(ks,es,'k--')
ax1.text(5e-4,2.2e6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.text(2e-3,1.3e4,r'$k^{-4/3}$')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax1.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(423)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.fill_between(kr*1e3,env_down_Ediv*1e-3/np.pi, env_up_Ediv*1e-3/np.pi, color='tab:green', alpha = 0.5, label = r'$\chi$')
ax1.fill_between(kr*1e3,env_down_Evort*1e-3/np.pi, env_up_Evort*1e-3/np.pi, color='tab:blue', alpha = 0.5, label = r'$\zeta$')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(frameon=False)
ks = np.array([2.5e-4,5.2e-4])
es = 9e-11*ks**(-1.1)
ax1.loglog(ks,es,'k--')
ax1.text(4e-4,7e-7,r'$k^{-1}$',fontsize=18)
ks = np.array([7e-4,5e-3])
es = 3.3e-5*ks**(2/3)
ax1.loglog(ks,es,'k--')
ax1.text(1.4e-3,6.5e-7,r'$k^{2/3}$',fontsize=18)
# ax1.set_ylim((5e-9,1.5e-6))
ax1.text(5e-4,1.95e-6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(424)
ax2 = ax1.twiny()
ax1.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.fill_between(kr*1e3,env_down_Re, env_up_Re, color='grey', alpha = 0.5)
ax1.set_xscale('log')
ax1.axhline(y=1,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(425)
ax2 = ax1.twiny()
ax1.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# ax1.semilogx(kr*1e3,Qe_MAT[:,0]*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.fill_between(kr*1e3,env_down_Qe*1e10, env_up_Qe*1e10, color='grey', alpha = 0.5)
ax1.set_xscale('log')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(426)
ax2 = ax1.twiny()
ax1.annotate("e", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# ax1.semilogx(kr*1e3,Tpi_MAT[:,0]*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.fill_between(kr*1e3,env_down_Tpi*1e12, env_up_Tpi*1e12, color='grey', alpha = 0.5)
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(427)
ax2 = ax1.twiny()
ax1.annotate("f", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,c_50*1e10, lw=3,c= 'tab:green',label = r'$50 \; \mathrm{km}$')
ax1.semilogx(kr*1e3,c_100*1e10, lw=3,c= 'tab:orange',label = r'$150 \; \mathrm{km}$')
ax1.semilogx(kr*1e3,MAT_qe[2,:].T*1e10, lw=3,c= 'tab:blue',label = r'$250 \; \mathrm{km}$')
ax1.semilogx(kr*1e3,c_500*1e10, lw=3,c= 'tab:red',label = r'$500 \; \mathrm{km}$')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.set_xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
ax1.legend(frameon=False)
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')


ax1 = fig.add_subplot(428)
ax2 = ax1.twiny()
ax1.annotate("g", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,tpi_50*1e12,lw = 3,c = 'tab:green')#, label = '50 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.semilogx(kr*1e3,tpi_100*1e12,lw = 3 ,c = 'tab:orange')#, label = '150 km')#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.semilogx(kr*1e3,MAT_tpi[2,:].T*1e12, lw = 3,c = 'tab:blue')#, lw = 3, label = '250 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.semilogx(kr*1e3,tpi_500*1e12,lw = 3,c = 'tab:red')#, label = '500 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

plt.tight_layout()

fig.savefig(dir_out+'noisy_field_sensitivity.pdf', bbox_inches='tight')
plt.close(fig)



## Extended Data Figure 7-8 - same but with n0204

u_load = np.load(dir+'n0204/u_n0204.npy') # 10km/pixel data for n0204
v_load = np.load(dir+'n0204/v_n0204.npy') # 10km/pixel data
im = np.load(dir+'n0204/im_n03.npy') # 10km/pixel data

## get rids of the zeros - otherwise causes problem in the spectra
list = np.argwhere(im==0)
for i in range(np.argwhere(im==0).shape[0]):
    im[list[i][0],list[i][1]]=im[list[i][0],list[i][1]-1]
np.sum(im==0)

I_0 = np.max(im)
Tau = np.log10(I_0/im)

## make the data doubly periodic ##
tau_per = jf.doubly_per(Tau)
u_per = jf.doubly_per(u_load)
v_per = jf.doubly_per(v_load)

## detrend the data ##
Tau = jf.detrend_data(Tau, detrend='Both')
u_l = jf.detrend_data(u_load, detrend='Both')
v_l = jf.detrend_data(v_load, detrend='Both')

tau = jf.detrend_data(tau_per, detrend='Both')
u = jf.detrend_data(u_per, detrend='Both')
v = jf.detrend_data(v_per, detrend='Both')

alpha = 9500*N0**2/g
PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N


n = 1
wc = 200 # 200 for n0204

## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
## retrieve psi and co from u and v ##
PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

# for n0204
kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)
kr , E_div = jf.wv_spec(DIV_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)

# for n0204
u_l = u_l[1:,1:]
v_l = v_l[1:,1:]
spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
KE = 0.5*(E_u+E_v)

kr , E_tau = jf.wv_spec(Tau[1:,1:],dx,dy,detrend=None) # for n0204

T = tau
Nj,Ni = T.shape
_,wavnum2D,kx,ky = jf.wvnumb_vector(dx,dy,Ni,Nj)
spec_tk = np.fft.fft2(T)*wavnum2D*2*np.pi
TK = np.fft.ifft2(spec_tk).real

kr , E_tk = jf.wv_spec(TK[im.shape[0]+1:,1:im.shape[1]],dx,dy,detrend=None) # for n0204

kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)
# kr,tpi,Eke_prod = jf.KE_flux_doubly_per(filt_u,filt_v,dx,dy,detrend=None) # at some point with Patrice we explored in double periodic and setting some part to 0 but I am not sure I did it properly - should probably look more into the symmetry of the streamfunction to see if I did u = -u correctly or not .... Maybe something for follow up paper...

### running mean
VAR = Eke_prod*kr
N = 17 # moving average window N = 10 corresponds to 1 km
Q_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    Q_e = c.squeeze()
Q_e[0]=Eke_prod[0]*kr[0]
Q_e[-1]=Eke_prod[-1]*kr[-1]

VAR = E_div/E_vort
N=5 # moving average window N = 10 corresponds to 1 km
R_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    R_e = c.squeeze()
R_e[0]=E_div[0]/E_vort[0]
R_e[-1]=E_div[-1]/E_vort[-1]

kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend='Both') # for n0204

lim = 1 ## limit for imshow
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
plt.imshow(Tau, origin='lower',cmap ='bone', vmin = -0.7, vmax = 1.)
plt.colorbar(shrink = 0.4, label = r'$\tau$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta, k < 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)
V = VORT_b[im.shape[0]:,:im.shape[1],1].real
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
plt.imshow(V_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.colorbar(shrink = 0.4, label = r'$\zeta$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}, k > 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)
TK = VORT_tau[im.shape[0]:,:im.shape[1],1].real
spec_thp, TK_filt = jf.cut_hp(dx,dy,TK,6e-4*1e-3)
plt.imshow(TK_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -3.5, vmax = 3.5)
plt.colorbar(shrink = 0.4, label = r'$\zeta_{\tau}$ $[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()

fig.savefig(dir_out+'Extended_data_Fig_lim_'+str(lim)+'_n0204.pdf', bbox_inches='tight')
plt.close(fig)


## Modified figure spectra according to the reviewer 3 recommendation
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(26,26))

ax1 = fig.add_subplot(321)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-3/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{APE}$') #label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
ax1.loglog(kr*1e3,KE*1e-3/np.pi, lw=3, label = r'$\mathrm{KE}$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1],  labels[0]]
ax1.legend(handles,labels,frameon=False,loc=3)
ks = np.array([2.5e-4,5.2e-4])
es = 6e-6*ks**-3
ax1.loglog(ks,es,'k--')
ax1.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
es = 1.1e0*ks**-(4/3)
ax1.loglog(ks,es,'k--')
ax1.text(5e-4,2.2e6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.text(2e-3,1.3e4,r'$k^{-4/3}$')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax1.set_ylim((1e1,1.3e6))
ax1.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(323)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-3/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
ax1.loglog(kr*1e3,E_div*1e-3/np.pi, lw=3, label = r'$\chi$', c='tab:green')
ax1.loglog(kr*1e3,E_vort*1e-3/np.pi, lw=3, label = r'$\zeta$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
ax1.legend(handles,labels,frameon=False)
ks = np.array([2.5e-4,5.2e-4])
es = 9e-11*ks**(-1.1)
ax1.loglog(ks,es,'k--')
ax1.text(4e-4,7e-7,r'$k^{-1}$',fontsize=18)
ks = np.array([7e-4,5e-3])
es = 3.3e-5*ks**(2/3)
ax1.loglog(ks,es,'k--')
ax1.text(1.4e-3,6.5e-7,r'$k^{2/3}$',fontsize=18)
ax1.set_ylim((5e-9,1.5e-6))
ax1.text(5e-4,1.95e-6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(324)
ax2 = ax1.twiny()
ax1.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.axhline(y=1,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(325)
ax2 = ax1.twiny()
ax1.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(326)
ax2 = ax1.twiny()
ax1.annotate("e", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')


plt.tight_layout()

fig.savefig(dir_out+'Extended_data_fig_n0204_n1_cutoff_200_H0_9500.pdf', bbox_inches='tight')


## figure 4 KE and ens transfer
## enstrophy flux ##
kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')
# kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend='Both') # for n0204


plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(121)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(122)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

plt.tight_layout()


##########################################################################################


## make contour for the polygon
coord_f2 = [[1280,170], [1500,650], [1110,650], [900,170]]
coord_f2.append(coord_f2[0]) #repeat the first point to create a 'closed loop'
xs_f2, ys_f2 = zip(*coord_f2)


## Plot fields in the physical space
lim = 1.5 ## limit for imshow
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
plt.plot(ys_f2,xs_f2, lw=5, c='k')


plt.imshow(Tau, origin='lower',cmap ='bone', vmin = -0.7, vmax = 1.)
plt.colorbar(shrink = 0.4)
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)
plt.imshow(VORT_b[im.shape[0]:,:im.shape[1],1].real*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
plt.imshow(DIV_b[im.shape[0]:,:im.shape[1],1].real*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()


fig.savefig(dir_out+'Fig1_lim_'+str(lim)+'.png', bbox_inches='tight')
plt.close(fig)


## Extended Data


## Figure 2
# n = 2 ## these parameters (n and wc) were explored at some point with Tricia's wind field at 15 km/pixel
# wc = 150
n = 1
wc = 250 # 200 for n0204

## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
## retrieve psi and co from u and v ##
PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
kr , E_div = jf.wv_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
# for n0204
kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)
kr , E_div = jf.wv_spec(DIV_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend=None)

spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

# for n0204
# u_l = u_l[1:,1:]
# v_l = v_l[1:,1:]
# spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
# spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
KE = 0.5*(E_u+E_v)

kr , E_tau = jf.wv_spec(Tau,dx,dy,detrend=None)
# kr , E_tau = jf.wv_spec(Tau[1:,1:],dx,dy,detrend=None) # for n0204


T = tau
Nj,Ni = T.shape
_,wavnum2D,kx,ky = jf.wvnumb_vector(dx,dy,Ni,Nj)
spec_tk = np.fft.fft2(T)*wavnum2D*2*np.pi
TK = np.fft.ifft2(spec_tk).real

kr , E_tk = jf.wv_spec(TK[im.shape[0]:,:im.shape[1]],dx,dy,detrend=None)
# kr , E_tk = jf.wv_spec(TK[im.shape[0]+1:,1:im.shape[1]],dx,dy,detrend=None) # for n0204


kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)
# kr,tpi,Eke_prod = jf.KE_flux_doubly_per(filt_u,filt_v,dx,dy,detrend=None) # at some point with Patrice we explored in double periodic and setting some part to 0 but I am not sure I did it properly - should probably look more into the symmetry of the streamfunction to see if I did u = -u correctly or not .... Maybe something for follow up paper...


### running mean
VAR = Eke_prod*kr
N= 17 # moving average window N = 10 corresponds to 1 km
Q_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    Q_e = c.squeeze()
Q_e[0]=Eke_prod[0]*kr[0]
Q_e[-1]=Eke_prod[-1]*kr[-1]

VAR = E_div/E_vort
N=5 # moving average window N = 10 corresponds to 1 km
R_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    R_e = c.squeeze()
R_e[0]=E_div[0]/E_vort[0]
R_e[-1]=E_div[-1]/E_vort[-1]

## some old piece of code to compute spectra - probably to compare to jf.wv_spec function

# VORT_b = jf.detrend_data(VORT_b[:,:,1].real, detrend='Both')
# VO = spec.TWODimensional_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy, detrend = False)
# kr, E_vort = VO.ki, VO.ispec
#
# DI = spec.TWODimensional_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy, detrend = False)
# kr, E_vort = DI.ki, DI.ispec

# U = spec.TWODimensional_spec(filt_u,dx,dy, detrend = False)
# kr, E_u = U.ki, U.ispec

# V = spec.TWODimensional_spec(filt_v,dx,dy, detrend = False)
# kr, E_v = V.ki, V.ispec

# kr , E_taun = jf.wv_spec(tau[im.shape[0]:,:im.shape[1]],dx,dy,detrend=None)
# TAU = spec.TWODimensional_spec(Tau,dx,dy, detrend = False)
# kr , E_tau  = TAU.ki, TAU.ispec

# TAU_K = spec.TWODimensional_spec(TK[im.shape[0]:,:im.shape[1]],dx,dy, detrend = False)
# kr , E_tk = TAU_K.ki, TAU_K.ispec

## Modified figure spectra according to the reviewer 3 recommendation
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(12,24))

ax1 = fig.add_subplot(311)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-3/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{APE}$') #label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
ax1.loglog(kr*1e3,KE*1e-3/np.pi, lw=3, label = r'$\mathrm{KE}$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1],  labels[0]]
ax1.legend(handles,labels,frameon=False,loc=3)
ks = np.array([2.5e-4,5.2e-4])
es = 6e-6*ks**-3
ax1.loglog(ks,es,'k--')
ax1.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
es = 1.1e0*ks**-(4/3)
ax1.loglog(ks,es,'k--')
ax1.text(5e-4,2.2e6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.text(2e-3,1.3e4,r'$k^{-4/3}$')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax1.set_ylim((1e1,1.3e6))
ax1.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(312)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-3/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
ax1.loglog(kr*1e3,E_div*1e-3/np.pi, lw=3, label = r'$\chi$', c='tab:green')
ax1.loglog(kr*1e3,E_vort*1e-3/np.pi, lw=3, label = r'$\zeta$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
ax1.legend(handles,labels,frameon=False)
ks = np.array([2.5e-4,5.2e-4])
es = 9e-11*ks**(-1.1)
ax1.loglog(ks,es,'k--')
ax1.text(4e-4,7e-7,r'$k^{-1}$',fontsize=18)
ks = np.array([7e-4,5e-3])
es = 3.3e-5*ks**(2/3)
ax1.loglog(ks,es,'k--')
ax1.text(1.4e-3,6.5e-7,r'$k^{2/3}$',fontsize=18)
ax1.set_ylim((5e-9,1.5e-6))
ax1.text(5e-4,1.95e-6,r'$1600$')
ax1.axvline(x = 6e-4,ls ='dotted', c='k')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(313)
ax2 = ax1.twiny()
ax1.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.axhline(y=1,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')
plt.tight_layout()

fig.savefig(dir_out+'Figure2_n0204_n1_cutoff_200_H0_9500.png', bbox_inches='tight')


## new figure 4 KE and ens transfer


## enstrophy flux ##
kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')
# kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]+1:,1:im.shape[1],1].real,dx,dy,detrend='Both') # for n0204


plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(121)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(122)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')

ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

plt.tight_layout()

fig.savefig(dir_out+'Figure4_n0204_n1_cutoff_200_H0_9500.png', bbox_inches='tight')

## figure 2 with wavelength as well as wavenumber
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,16))

ax1 = fig.add_subplot(221)
ax2 = ax1.twiny()
ax1.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-3/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{APE}$') #label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
ax1.loglog(kr*1e3,KE*1e-3/np.pi, lw=3, label = r'$\mathrm{KE}$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1],  labels[0]]
ax1.legend(handles,labels,frameon=False,loc=3)
ks = np.array([2.5e-4,5.6e-4])
es = 6e-6*ks**-3
ax1.loglog(ks,es,'k--')
ax1.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
es = 1.1e0*ks**-(4/3)
ax1.loglog(ks,es,'k--')
ax1.text(2e-3,1.3e4,r'$k^{-4/3}$')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax1.set_ylim((1e1,1e6))
ax1.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(222)
ax2 = ax1.twiny()
ax1.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-3/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
ax1.loglog(kr*1e3,E_div*1e-3/np.pi, lw=3, label = r'$\chi$', c='tab:green')
ax1.loglog(kr*1e3,E_vort*1e-3/np.pi, lw=3, label = r'$\zeta$')
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
ax1.legend(handles,labels,frameon=False)
ks = np.array([2.5e-4,6e-4])
es = 9e-11*ks**(-1.1)
ax1.loglog(ks,es,'k--')
ax1.text(4e-4,7e-7,r'$k^{-1}$',fontsize=18)
ks = np.array([7e-4,5e-3])
es = 3.3e-5*ks**(2/3)
ax1.loglog(ks,es,'k--')
ax1.text(1.4e-3,6.5e-7,r'$k^{2/3}$',fontsize=18)
ax1.set_ylim((5e-9,1.5e-6))
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(223)
ax2 = ax1.twiny()
ax1.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
ax1.axhline(y=1,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')

ax1 = fig.add_subplot(224)
ax2 = ax1.twiny()
ax1.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
ax1.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax1.axhline(y=0,c='gray', ls='dashed')
ax1.set_xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
ax1.set_ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
ax2.invert_xaxis()
ax2.set_xscale('log')
ax2.set_xlim((1/ax1.get_xlim()[0],1/ax1.get_xlim()[1]))
ax2.set_xlabel('$\mathrm{wavelength} \; \mathrm{[km]}$')
plt.tight_layout()

fig.savefig(dir_out+'Figure2.png', bbox_inches='tight')



## Figure 2 old - without the wavenumber on top
plt.rcParams["text.usetex"] = True
## plot spectra
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(24,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-3/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{APE}$')
# plt.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-6/np.pi, lw=3,c='tab:orange', label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
plt.loglog(kr*1e3,KE*1e-3/np.pi, lw=3, label = r'$\mathrm{KE}$')
handles,labels = plt.gca().get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1],  labels[0]]
plt.legend(handles,labels,frameon=False,loc=3)
ks = np.array([2.5e-4,5.6e-4])
es = 6e-6*ks**-3
plt.loglog(ks,es,'k--')
plt.text(3.6e-4,2e5,r'$k^{-3}$',fontsize=18)
ks = np.array([7e-4,8e-3])
# es = 1.3e-4*ks**-(5/3)
# plt.loglog(ks,es,'k--')
# plt.text(2e-3,1.3e1,r'$k^{-5/3}$')
es = 1.1e0*ks**-(4/3)
plt.loglog(ks,es,'k--')
plt.text(2e-3,1.3e4,r'$k^{-4/3}$')
# plt.grid()
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel('$\mathrm{PSD}$ '+r'$\mathrm{[m}^{2}\mathrm{s}^{-2}/\mathrm{cpkm]}$')
plt.ylim((1e1,1e6))
plt.annotate("$1600$ $\mathrm{km}$", xy=(0.31, 1.015), xycoords="axes fraction")
plt.axvline(x=6e-4,c='gray', ls='dashed')

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-3/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
plt.loglog(kr*1e3,E_div*1e-3/np.pi, lw=3, label = r'$\chi$', c='tab:green')
plt.loglog(kr*1e3,E_vort*1e-3/np.pi, lw=3, label = r'$\zeta$')
handles,labels = plt.gca().get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
plt.legend(handles,labels,frameon=False)
plt.annotate("$1600$ $\mathrm{km}$", xy=(0.31, 1.015), xycoords="axes fraction")
plt.axvline(x=6e-4,c='gray', ls='dashed')
# plt.grid()
plt.ylim((5e-9,1e-6))
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
# plt.ylabel('PSD '+r'[s$^{-2}$/cpm]')
plt.ylabel('$\mathrm{PSD}$ '+r'$[\mathrm{s}^{-2}/\mathrm{cpkm]}$')

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.axhline(y=1,c='gray', ls='dashed')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
## for pi_ke
# plt.semilogx(kr*1e3,tpi*1e4,lw = 3 )
# plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
# plt.ylabel(r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.axhline(y=0,c='gray', ls='dashed')
## for k.KE_adv
plt.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.tight_layout()
fig.savefig(dir_out+'Figure2.png', bbox_inches='tight')

## old 3 figures panels ##
# plt.rcParams["text.usetex"] = True
# ## plot spectra
# plt.rcParams['font.size'] = '22'
# fig = plt.figure(figsize=(32,7))
# plt.subplot(131)
# plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-6/np.pi, lw=3,c='tab:orange', label = r'$\mathrm{PE}$')
# # plt.loglog(kr*1e3,E_tau*(g*alpha/N0)**2*1e-6/np.pi, lw=3,c='tab:orange', label = r'$\big(\mathrm{H}_0 \: N \; \tau \big) ^2$')
# plt.loglog(kr*1e3,KE*1e-6/np.pi, lw=3, label = r'$\mathrm{KE}$')
# handles,labels = plt.gca().get_legend_handles_labels()
# handles = [handles[1], handles[0]]
# labels = [labels[1],  labels[0]]
# plt.legend(handles,labels,frameon=False,loc=3)
# ks = np.array([2.5e-4,5.6e-4])
# es = 6e-9*ks**-3
# plt.loglog(ks,es,'k--')
# plt.text(3.6e-4,2e2,r'$k^{-3}$',fontsize=18)
# ks = np.array([7e-4,8e-3])
# # es = 1.3e-4*ks**-(5/3)
# # plt.loglog(ks,es,'k--')
# # plt.text(2e-3,1.3e1,r'$k^{-5/3}$')
# es = 1.1e-3*ks**-(4/3)
# plt.loglog(ks,es,'k--')
# plt.text(2e-3,1.3e1,r'$k^{-4/3}$')
# # plt.grid()
# plt.xlabel('wavenumber [cpkm]')
# plt.ylabel('PSD '+r'[m$^{2}$s$^{-2}$/cpm]')
# plt.ylim((1e-2,1e3))
# plt.annotate("1600 km", xy=(0.31, 1.015), xycoords="axes fraction")
# plt.axvline(x=6e-4,c='gray', ls='dashed')
#
# plt.subplot(132)
# plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-6/np.pi, lw=3, label = r'$\zeta_{\tau}$',c = 'tab:orange')
# plt.loglog(kr*1e3,E_div*1e-6/np.pi, lw=3, label = r'$\chi$', c='tab:green')
# plt.loglog(kr*1e3,E_vort*1e-6/np.pi, lw=3, label = r'$\zeta$')
# # plt.loglog(kr*1e3,(g*alpha/N0)**2*E_tk*1e-6/np.pi, lw=3, label = r'$\big(k \: \mathrm{H}_0 \: N \: \tau \big)^2 $',c = 'tab:orange')
# # plt.loglog(kr*1e3,E_div*1e-6/np.pi, lw=3, label = r'$\chi^2$', c='tab:green')
# # plt.loglog(kr*1e3,E_vort*1e-6/np.pi, lw=3, label = r'$\zeta^2$')
# #
# # ks = np.array([7e-4,9e-3])
# # es = 1.3e-7*ks**(3/3)
# # plt.loglog(ks,es,'k--')
#
#
# handles,labels = plt.gca().get_legend_handles_labels()
# handles = [handles[2], handles[1], handles[0]]
# labels = [labels[2], labels[1], labels[0]]
# plt.legend(handles,labels,frameon=False)
# plt.annotate("1600 km", xy=(0.31, 1.015), xycoords="axes fraction")
# plt.axvline(x=6e-4,c='gray', ls='dashed')
# # plt.grid()
# plt.ylim((5e-12,1e-9))
# plt.xlabel('wavenumber [cpkm]')
# plt.ylabel('PSD '+r'[s$^{-2}$/cpm]')
#
# plt.subplot(133)
# plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.semilogx(kr*1e3,Q_e*7e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]',c='tab:orange')
# plt.semilogx(kr*1e3,tpi*1e4,lw = 3 , label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
#
# handles,labels = plt.gca().get_legend_handles_labels()
# handles = [handles[1], handles[0]]
# labels = [labels[1], labels[0]]
# plt.legend(handles,labels,frameon=False,loc = 3)#, fontsize=16)
# # plt.grid()
# # plt.gca().axvspan(1/1500, 1/500, alpha=0.3, color='gray')
# plt.xlabel('wavenumber [cpkm]')
# # plt.ylabel(r'[m$^{2}$s$^{-3}$]')
# # plt.annotate("210 km", xy=(0.59, 1.015), xycoords="axes fraction")
# # plt.axvline(x=1/210,c='gray', ls='dashed')
# plt.axhline(y=0,c='gray', ls='dashed')
# plt.tight_layout()
#
# fig.savefig(dir_out+'spec_4_3_nogrid.png', bbox_inches='tight')


## enstrophy flux ##
kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')
kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(u_l,v_l,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')

plt.rcParams['font.size'] = '20'
fig = plt.figure(figsize=(11,7))
plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.semilogx(kr*1e3,Eke_prod_vort*kr*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.title(r'$\Pi_{\mathrm{ENS}}$ ', fontsize = 30)
# plt.grid()
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
# plt.ylabel(r'[$10^{12}$ $\mathrm{m}\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.tight_layout()
fig.savefig(dir_out+'flux_enstrophy_nogrid.png', bbox_inches='tight')




VAR = E_div/E_vort
N=5 # moving average window N = 10 corresponds to 1 km
R_e = np.empty((VAR.shape[0]))*np.nan
for j in range(len(VAR)):
    bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
    c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
    R_e = c.squeeze()
R_e[0]=E_div[0]/E_vort[0]
R_e[-1]=E_div[-1]/E_vort[-1]



## ratio of zeta / div
plt.rcParams['font.size'] = '20'
fig = plt.figure(figsize=(11,7))
# plt.semilogx(kr*1e3,E_div/E_vort,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,R_e,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.axhline(y=1,c='gray', ls='dashed')
plt.title(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$', fontsize = 30)
# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')
plt.tight_layout()
fig.savefig(dir_out+'ratio_zeta_vort_nogrid.png', bbox_inches='tight')



## combination of ratio div/zeta and enstrophy

plt.rcParams['font.size'] = '20'
fig = plt.figure(figsize=(22,7))
plt.subplot(121)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
plt.semilogx(kr*1e3,R_e,lw = 3 )
plt.axhline(y=1,c='gray', ls='dashed')
plt.title(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$', fontsize = 30)
# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$')

plt.subplot(122)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.semilogx(kr*1e3,Eke_prod_vort*kr*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.title(r'$\Pi_{\mathrm{ENS}}$ ', fontsize = 30)
# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
# plt.ylabel(r'[$10^{12}$ $\mathrm{m}\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')


plt.tight_layout()

fig.savefig(dir_out+'ratio_zeta_vort_enstr_nogrid.png', bbox_inches='tight')

## combination of ratio KE_flux, k.KE_adv and enstrophy

plt.rcParams['font.size'] = '20'
fig = plt.figure(figsize=(22,5))
plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
plt.semilogx(kr*1e3,tpi*1e3,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.semilogx(kr*1e3,Eke_prod_vort*kr*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.title(r'$\Pi_{\mathrm{KE}}$ ', fontsize = 30)

# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$\Pi_{\mathrm{KE}}$ [$10^{3}$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.ylabel(r'[$10^{12}$ $\mathrm{m}\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,Q_e*1e9,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.title(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ ', fontsize = 30)
# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{9}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')

# plt.ylabel(r'[$10^{12}$ $\mathrm{m}\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')

plt.subplot(133)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
# plt.semilogx(kr*1e3,Eke_prod_vort*kr*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.title(r'$\Pi_{\mathrm{ENS}}$ ', fontsize = 30)
# plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
# plt.ylabel(r'[$10^{12}$ $\mathrm{m}\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')


plt.tight_layout()

fig.savefig(dir_out+'fluxes_nofilter.png', bbox_inches='tight')

## combination of ratio div/zeta and enstrophy - 1 plot

plt.rcParams['font.size'] = '24'
fig = plt.figure(figsize=(22,7))
plt.subplot(121)
plt.semilogx(kr*1e3,R_e,lw = 3 ,label =r'$\chi_{\mathrm{PSD}}/\zeta_{\mathrm{PSD}}$' )
plt.axhline(y=1,c='gray', ls='dashed')
plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 , label = r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
plt.grid()
plt.xlabel('wavenumber [cpkm]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.legend(frameon=False, fontsize = 24)
plt.tight_layout()

fig.savefig(dir_out+'ratio_zeta_vort_enstr_comb.png', bbox_inches='tight')


### figure sup with k.KE_adv and enstrophy flux
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(22,7))
plt.subplot(121)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$k \: \mathrm{KE}_{\mathrm{adv}}$', fontsize=30)
plt.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')

plt.subplot(122)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$\Pi_{\mathrm{ENS}}$', fontsize=30)
plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')


plt.tight_layout()

fig.savefig(dir_out+'k_ke_adv_enstrophy_b.png', bbox_inches='tight')

## explore sensitivity of the cascade to the processing filtering
n = 1
wc = 250
n_list = [1,2,3,4,5]
n_list = [1]
wc_list= [50,100,250,500]
MAT_tpi = np.empty((len(wc_list)*len(n_list), 369))
MAT_qe = np.empty((len(wc_list)*len(n_list), 369))
I=0
for n in n_list:
    print(n)
    for wc in wc_list:
        print(wc)
        print("I = "+str(I))
        ## apply a butterworth low pass to u and v ##
        spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
        spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
        ## retrieve psi and co from u and v ##
        PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

        kr , E_vort = jf.wv_spec(VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)
        kr , E_div = jf.wv_spec(DIV_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend=None)

        spec_u, filt_u = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
        spec_v, filt_v = jf.butterworh_lp(dx,dy,v_l,n,wl_c=wc)

        kr, E_u= jf.wv_spec(filt_u,dx,dy,detrend=None)
        kr, E_v= jf.wv_spec(filt_v,dx,dy,detrend=None)
        KE = 0.5*(E_u+E_v)

        kr,tpi,Eke_prod = jf.KE_flux(filt_u,filt_v,dx,dy,detrend=None)
        kr,tpi_vort,Eke_prod_vort = jf.tracer_flux(filt_u,filt_v,VORT_b[im.shape[0]:,:im.shape[1],1].real,dx,dy,detrend='Both')


        ### running mean
        VAR = Eke_prod*kr
        N= 17 # moving average window N = 10 corresponds to 1 km
        Q_e = np.empty((VAR.shape[0]))*np.nan
        for j in range(len(VAR)):
            bb = np.concatenate((np.repeat(VAR[0],N),VAR.squeeze(),np.repeat(VAR[-1],N))).squeeze()
            c = np.convolve(bb,np.ones((N,))/N)[int(np.floor((N-1)+N/2-1/2)):int(np.floor(-N-0.5-N/2))]
            Q_e = c.squeeze()
        Q_e[0]=Eke_prod[0]*kr[0]
        Q_e[-1]=Eke_prod[-1]*kr[-1]

        MAT_qe[I,:] =  Q_e
        MAT_tpi[I,:] =  tpi_vort
        # ### figure sup with k.KE_adv and enstrophy flux
        # plt.rcParams['font.size'] = '22'
        # fig = plt.figure(figsize=(22,7))
        # plt.subplot(121)
        # plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
        # # plt.title(r'$k \: \mathrm{KE}_{\mathrm{adv}}$', fontsize=30)
        # plt.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
        # plt.axhline(y=0,c='gray', ls='dashed')
        # plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
        # plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
        # plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
        # plt.subplot(122)
        # plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
        # # plt.title(r'$\Pi_{\mathrm{ENS}}$', fontsize=30)
        # plt.semilogx(kr*1e3,tpi_vort*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
        # plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
        # plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
        # plt.axhline(y=0,c='gray', ls='dashed')
        # plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
        #
        # plt.tight_layout()
        # fig.savefig(dir_out+'k_ke_adv_enstrophy_n_'+str(n)+'_wc_'+str(wc)+'.png', bbox_inches='tight')
        # plt.close(fig)

        I=I+1

# normalize matrix per respect to curve in main manuscript

c_50 = MAT_qe[0,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[0,:]))
c_100 = MAT_qe[1,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[1,:]))
c_500 = MAT_qe[-1,:]*np.max(np.abs(MAT_qe[2,:]))/np.max(np.abs(MAT_qe[-1,:]))

tpi_50 = MAT_tpi[0,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[0,:]))
tpi_100 = MAT_tpi[1,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[1,:]))
tpi_500 = MAT_tpi[-1,:]*np.max(np.abs(MAT_tpi[2,:]))/np.max(np.abs(MAT_tpi[-1,:]))


plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(22,7))
plt.subplot(121)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$k \: \mathrm{KE}_{\mathrm{adv}}$', fontsize=30)

plt.semilogx(kr*1e3,c_50*1e10, lw=3,c= 'tab:green',label = r'$50 \; \mathrm{km}$')
plt.semilogx(kr*1e3,c_100*1e10, lw=3,c= 'tab:orange',label = r'$100 \; \mathrm{km}$')
plt.semilogx(kr*1e3,MAT_qe[2,:].T*1e10, lw=3,c= 'tab:blue',label = r'$250 \; \mathrm{km}$')
plt.semilogx(kr*1e3,c_500*1e10, lw=3,c= 'tab:red',label = r'$500 \; \mathrm{km}$')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
plt.legend()

plt.subplot(122)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$\Pi_{\mathrm{ENS}}$', fontsize=30)

plt.semilogx(kr*1e3,tpi_50*1e12,lw = 3,c = 'tab:green')#, label = '50 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,tpi_100*1e12,lw = 3 ,c = 'tab:orange')#, label = '150 km')#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,MAT_tpi[2,:].T*1e12, lw = 3,c = 'tab:blue')#, lw = 3, label = '250 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,tpi_500*1e12,lw = 3,c = 'tab:red')#, label = '500 km' )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))


plt.tight_layout()
fig.savefig(dir_out+'k_ke_adv_enstrophy_normalized_bis.pdf', bbox_inches='tight')

# construc enveloppe
env_up_qe = np.zeros(MAT_qe.shape[1])
env_up_tpi = np.zeros(MAT_qe.shape[1])
env_down_qe = np.zeros(MAT_qe.shape[1])
env_down_tpi = np.zeros(MAT_qe.shape[1])
for i in range(MAT_qe.shape[1]):
    env_up_qe[i]=np.max(MAT_qe[:,i])
    env_up_tpi[i]=np.max(MAT_tpi[:,i])
    env_down_qe[i]=np.min(MAT_qe[:,i])
    env_down_tpi[i]=np.min(MAT_tpi[:,i])

numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)
std_up = np.std(MAT_qe, axis= 0)
std_mean = np.median(MAT_qe, axis= 0)
### figure sup with k.KE_adv and enstrophy flux
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(22,7))
plt.subplot(121)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$k \: \mathrm{KE}_{\mathrm{adv}}$', fontsize=30)
plt.semilogx(kr*1e3,Q_e*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
# plt.semilogx(kr*1e3,MAT_qe.T*1e10, lw=3,label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
# plt.semilogx(kr*1e3,env_down_qe*1e10, lw=4,c='black',label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
# plt.semilogx(kr*1e3,env_up_qe*1e10, lw=4,c='black',label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.semilogx(kr*1e3,std_mean*1e10+0.5*std_up*1e10, lw=4,c='black',label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.semilogx(kr*1e3,std_mean*1e10-0.5*std_up*1e10, lw=4,c='black',label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.semilogx(kr*1e3,std_mean*1e10, lw=4,c='black',label = r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$k \: \mathrm{KE}_{\mathrm{adv}}$ [$10^{10}$ $\mathrm{m}^{2}\mathrm{s}^{-3}\mathrm{cpm}$]')
plt.ylim((-5,5))
plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))
plt.subplot(122)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$\Pi_{\mathrm{ENS}}$', fontsize=30)
plt.semilogx(kr*1e3,MAT_tpi.T*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,env_down_tpi*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.semilogx(kr*1e3,env_up_tpi*1e12,lw = 3 )#, label = r'$\Pi_{\mathrm{KE}}$ [$10^4$ $\mathrm{m}^{2}\mathrm{s}^{-3}$]')
plt.xlabel('$\mathrm{wavenumber} \; \mathrm{[cpkm]}$')
plt.ylabel(r'$\Pi_{\mathrm{ENS}}$ [$10^{12}$ $\mathrm{s}^{-3}$]')
plt.axhline(y=0,c='gray', ls='dashed')
plt.xlim((min(kr*1e3)-1.5e-5,max(kr*1e3)+2e-2))

plt.tight_layout()

fig.savefig(dir_out+'k_ke_adv_enstrophy_all.png', bbox_inches='tight')
plt.close(fig)







## vorticity distribution ##
V = VORT_b[u_load.shape[0]:,:u_load.shape[1],1].real
from scipy.stats import skew


fig = plt.figure(figsize=(20, h+5))

# plt.subplots_adjust(wspace= 1, hspace= 0.25)

sub1 = fig.add_subplot(4,3,1) # two rows, two columns, fist cell
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# spec_thp, V_filt = jf.cut_lp(dx,dy,V,(1/250)*1e-3)
S=skew(V.flatten())
n, bins, patches = plt.hist(V.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
plt.title('Full field')
plt.xlabel(r'$\zeta$ [$10^4$ s$^{-1}$]')
plt.ylabel(r'occurence')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-15,15))
plt.ylim((1e2,2.5e5))

# Create second axes, the top-left plot with orange plot
sub2 = fig.add_subplot(4,3,2) # two rows, two columns, second cell
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k < 6.10^{-4}$ cpkm')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta$ [$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-2,2))
plt.ylim((1e2,2.5e5))

# Create second axes, the top-left plot with orange plot
sub3 = fig.add_subplot(4,3,3) # two rows, two columns, second cell
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
# spec_thp, V_filt = jf.cut_bp(dx,dy,V,(1/250)*1e-3,(1/2500)*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k > 6.10^{-4}$ cpkm')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta$ [$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-15,15))
plt.ylim((1e2,2.5e5))

# Create third axes, a combination of third and fourth cell
sub4 = fig.add_subplot(4,3,(4,10)) # two rows, two colums, combined third and fourth cell
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# plt.title(r'$\zeta$', fontsize = 34)
plt.imshow(V*1e4, origin='lower', cmap = 'RdBu_r', vmin = -2, vmax = 2)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$\zeta$ [$10^{4}$ s$^{-1}$]')
plt.xlabel(r'x [10 km]')
plt.ylabel('y [10 km]')




sub5 = fig.add_subplot(4,3,(5,11)) # two rows, two colums, combined third and fourth cell
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
plt.annotate("e", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# plt.title(r'$\zeta$', fontsize = 34)
plt.imshow(V_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -1.1, vmax = 1.1)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$\zeta$ [$10^{4}$ s$^{-1}$]')
plt.xlabel(r'x [10 km]')
# plt.ylabel('y [10 km]')


sub6 = fig.add_subplot(4,3,(6,12)) # two rows, two colums, combined third and fourth cell
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
plt.annotate("f", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# plt.title(r'$\zeta$', fontsize = 34)
plt.imshow(V_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -2, vmax = 2)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$\zeta$ [$10^{4}$ s$^{-1}$]')
plt.xlabel(r'x [10 km]')
# plt.ylabel('y [10 km]')

plt.tight_layout()

fig.savefig(dir_out+'vort_distribution.png', bbox_inches='tight')




### Rossby distribution ##
fig = plt.figure(figsize=(22, h+5))

# plt.subplots_adjust(wspace= 1, hspace= 0.25)

sub1 = fig.add_subplot(4,3,1) # two rows, two columns, fist cell
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# spec_thp, V_filt = jf.cut_lp(dx,dy,V,(1/250)*1e-3)
S=skew(V.flatten())
n, bins, patches = plt.hist(V.flatten()*1e4/3.4, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.63, 0.83), xycoords="axes fraction")
plt.title('$\mathrm{Full} \: \mathrm{field}$')
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,2.5e5))

# Create second axes, the top-left plot with orange plot
sub2 = fig.add_subplot(4,3,2) # two rows, two columns, second cell
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k < 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.63, 0.83), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta/f$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-0.6,0.6))
plt.ylim((1e2,2.5e5))

# Create second axes, the top-left plot with orange plot
sub3 = fig.add_subplot(4,3,3) # two rows, two columns, second cell
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
# spec_thp, V_filt = jf.cut_bp(dx,dy,V,(1/250)*1e-3,(1/2500)*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k > 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.63, 0.83), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta/f$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,2.5e5))

# Create third axes, a combination of third and fourth cell
sub4 = fig.add_subplot(4,3,(4,10)) # two rows, two colums, combined third and fourth cell
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
# plt.title(r'$\zeta$', fontsize = 34)
plt.imshow(V*1e4/3.4, origin='lower', cmap = 'RdBu_r', vmin = -1, vmax = 1)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.5, label = r'$\zeta /f$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')




sub5 = fig.add_subplot(4,3,(5,11)) # two rows, two colums, combined third and fourth cell
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
plt.annotate("e", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.imshow(V_filt*1e4/3.4, origin='lower', cmap = 'RdBu_r', vmin = -0.4, vmax = 0.4)
plt.colorbar(shrink = 0.5, label = r'$\zeta /f$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
# plt.ylabel('y [10 km]')


sub6 = fig.add_subplot(4,3,(6,12)) # two rows, two colums, combined third and fourth cell
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
plt.annotate("f", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.imshow(V_filt*1e4/3.4, origin='lower', cmap = 'RdBu_r', vmin = -1, vmax = 1)
plt.colorbar(shrink = 0.5, label = r'$\zeta /f$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
# plt.ylabel('y [10 km]')

plt.tight_layout()

fig.savefig(dir_out+'rossby_distribution.png', bbox_inches='tight')


### only the histograms
plt.rcParams['font.size'] = '18'
fig = plt.figure(figsize=(20,5+h))

plt.subplot(431)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# spec_thp, V_filt = jf.cut_lp(dx,dy,V,(1/250)*1e-3)
S=skew(V_filt.flatten())
n, bins, patches = plt.hist(V.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
plt.title('Full field')
plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.ylabel(r'occurence')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-15,15))
plt.ylim((1e2,2.5e5))

plt.subplot(432)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k < 6.10^{-4}$ cpkm')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4, 200, facecolor='tab:blue', alpha=0.75)
# n, bins, patches = plt.hist(V_filt.flatten()*1e4, 200, facecolor='tab:blue', alpha=0.75, weights=np.ones(len(V_filt.flatten())) / len(V_filt.flatten()),density = False)

plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-3,3))
plt.ylim((1e2,2.5e5))

plt.subplot(433)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
# spec_thp, V_filt = jf.cut_bp(dx,dy,V,(1/250)*1e-3,(1/2500)*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k > 6.10^{-4}$ cpkm')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-15,15))
plt.ylim((1e2,2.5e5))

plt.tight_layout()



## only the histograms in terms of Rossby number
V = VORT_b[u_load.shape[0]:,:u_load.shape[1],1].real
plt.rcParams['font.size'] = '18'
fig = plt.figure(figsize=(20,5))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# spec_thp, V_filt = jf.cut_lp(dx,dy,V,(1/250)*1e-3)
S=skew(V.flatten())
# S=skew(V_filt.flatten())
n, bins, patches = plt.hist(V.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
plt.title('$\mathrm{Full} \: \mathrm{field}$')
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,3e5))

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k < 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
# plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-0.6,0.6))
plt.ylim((1e2,3e5))
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
# plt.tight_layout()

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
spec_thp, V_filt = jf.cut_hp(dx,dy,V,6e-4*1e-3)
# spec_thp, V_filt = jf.cut_bp(dx,dy,V,(1/250)*1e-3,(1/2500)*1e-3)
S=skew(V_filt.flatten())
plt.title(r'$k > 6.10^{-4} \: \mathrm{cpkm}$')
plt.annotate(r'$\mathrm{skewness}$ = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(V_filt.flatten()*1e4/3.5, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\mathrm{occurence}$')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,3e5))

plt.tight_layout()
fig.savefig(dir_out+'rossby_distribution.pdf', bbox_inches='tight')


## new figure 1 (tau, vort ft large scale zeta_tau small scale )
## make contour for the polygon
coord_f2 = [[1280,170], [1500,650], [1110,650], [900,170]]
coord_f2.append(coord_f2[0]) #repeat the first point to create a 'closed loop'
xs_f2, ys_f2 = zip(*coord_f2)

lim = 1 ## limit for imshow
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')


plt.imshow(Tau, origin='lower',cmap ='bone', vmin = -0.7, vmax = 1.)
plt.colorbar(shrink = 0.4)
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta, k < 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)
spec_thp, V_filt = jf.cut_lp(dx,dy,V,6e-4*1e-3)
plt.imshow(V_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -lim, vmax = lim)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}, k > 6.10^{-4} \: \mathrm{cpkm}$', fontsize = 34)

TK = VORT_tau[im.shape[0]:,:im.shape[1],1].real
spec_thp, TK_filt = jf.cut_hp(dx,dy,TK,6e-4*1e-3)
# spec_thp, TK_filt = jf.cut_bp(dx,dy,TK, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
plt.imshow(TK_filt*1e4, origin='lower', cmap = 'RdBu_r', vmin = -3.5, vmax = 3.5)
# plt.plot(ys_f2,xs_f2, lw=5, c='k')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()


fig.savefig(dir_out+'new_Fig1_lim_'+str(lim)+'_n0204.png', bbox_inches='tight')
plt.close(fig)


## scatterplots of zeta_tau and zeta and divergence (Figure 3 of April 2nd 2021 submission)
V = VORT_b[im.shape[0]:,:im.shape[1],1].real
D = DIV_b[im.shape[0]:,:im.shape[1],1].real
TK = VORT_tau[im.shape[0]:,:im.shape[1],1].real
TAU = tau[im.shape[0]:,:im.shape[1]]


wl_low = 250
wl_high = (6e-4)

spec_thp, V_filt = jf.cut_bp(dx,dy,V, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, D_filt = jf.cut_bp(dx,dy,D, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, TK_filt = jf.cut_bp(dx,dy,TK, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)
spec_thp, TauK_filt = jf.cut_bp(dx,dy,TAU, wl_l = (1/wl_low)*1e-3,wl_h = (wl_high)*1e-3)


coord = [[600,789], [600,200], [1100,200], [1100,789]] ## coordinates of the polar vortex
coord.append(coord[0]) #repeat the first point to create a 'closed loop'
xs, ys = zip(*coord)


coord_f1 = [[0,260], [0,0], [300,0], [300,260]] ## coordinates of the filament in the lowerleft coner
coord_f1.append(coord_f1[0]) #repeat the first point to create a 'closed loop'
xs_f1, ys_f1 = zip(*coord_f1)


coord_f3 = [[1500,100], [2000,100], [2000,600], [1500,600]] ## coordinates of a region we dont use (I dont remember which region)
coord_f3.append(coord_f3[0]) #repeat the first point to create a 'closed loop'
xs_f3, ys_f3 = zip(*coord_f3)


coord_f2 = [[1280,170], [1500,650], [1110,650], [900,170]] ## coordinates of the famous streamer
coord_f2.append(coord_f2[0]) #repeat the first point to create a 'closed loop'
xs_f2, ys_f2 = zip(*coord_f2)


coord_f4 = [[1,2], [2078,2], [2078,789], [1,789]] ## coordinates of the full field
coord_f4.append(coord_f4[0]) #repeat the first point to create a 'closed loop'
xs_f4, ys_f4 = zip(*coord_f4)



var = V_filt
M_vort = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TK_filt[0:300,0:260]*1e4
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[600:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TK_filt[600:1100,200:-1]*1e4
    if j==2:
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]*1e4
    if j==3:
        Y = var*1e4 ## full field
        X = TK_filt*1e4

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    M_vort[:,0,j]=val_m
    M_vort[:,1,j]=val_my
    M_vort[:,2,j]=var_my

var = D_filt
M_div = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TK_filt[0:300,0:260]*1e4
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[500:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TK_filt[500:1100,200:-1]*1e4
    if j==2:
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]*1e4
    if j==3:
        Y = var*1e4 ## full field
        X = TK_filt*1e4

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    M_div[:,0,j]=val_m
    M_div[:,1,j]=val_my
    M_div[:,2,j]=var_my


## scatter plots between chi and tau - done during the revision stage
var = D_filt
T_div = np.zeros((199,3,4))
for j in range(4):
    print(j)
    if j==0:
        Y = var[0:300,0:260]*1e4
        X = TauK_filt[0:300,0:260]
    if j==1:
        # Y = var[500:1000,200:-1]*1e4 ## polar vortex - blue rectangle
        # X = -TK_filt[500:1000,200:-1]*1e4
        Y = var[500:1100,200:-1]*1e4 ## polar vortex - blue rectangle
        X = TauK_filt[500:1100,200:-1]
    if j==2:
        x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
        y = np.arange(0,V_filt.shape[0],1)
        # x = np.arange(0,790,1) # tried to mimic your data boundaries
        # y = np.arange(0,2080,1)
        xx, yy = np.meshgrid(x,y)
        m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

        Vc = var.copy()
        Vc[~m]=np.nan
        Tc = TauK_filt.copy()
        Tc[~m]=np.nan

        Y = Vc[np.isfinite(Vc)]*1e4
        X = Tc[np.isfinite(Vc)]
    if j==3:
        Y = var*1e4 ## full field
        X = TauK_filt

    x = X.flatten()
    y = Y.flatten()

    nbin=200
    #bin on x ##
    bin_x=np.linspace(np.nanmin(x),np.nanmax(x),nbin)

    ind_bin=np.empty((len(bin_x)-1,len(x)))
    ind_bin[:]=np.nan
    val_bin=np.empty((len(bin_x)-1,len(x)))
    val_bin[:]=np.nan

    val_y=np.empty((len(bin_x)-1,len(x)))
    val_y[:]=np.nan
    for i in range(len(bin_x)-1):
        z=np.squeeze(np.where((bin_x[i]<=x) & (x<bin_x[i+1])))
        l=z.size
        ind_bin[i,0:z.size]=z
        val_bin[i,0:z.size]=x[z]
        val_y[i,0:z.size]=y[z]
    val_m=np.nanmean(val_bin, axis=1)
    val_my=np.nanmean(val_y, axis=1)
    var_my=np.nanstd(val_y,axis=1)

    T_div[:,0,j]=val_m
    T_div[:,1,j]=val_my
    T_div[:,2,j]=var_my


###  histograms of zeta, div,  and zeta_tau
plt.rcParams['font.size'] = '18'
fig = plt.figure(figsize=(20,5+h))

plt.subplot(431)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
S=skew(V_filt.flatten())
n, bins, patches = plt.hist(V_filt.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
plt.title(r'$\zeta$')
plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.ylabel(r'occurence')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,1e5))

plt.subplot(432)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
S=skew(TK_filt.flatten())
plt.title(r'$\zeta_{\tau}$')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(TK_filt.flatten()*1e4, 200, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,1e5))

plt.subplot(433)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
S=skew(D_filt.flatten())
plt.title(r'$\chi$')
plt.annotate(r'skewness = '+str(np.round(S,2)), xy=(0.65, 0.85), xycoords="axes fraction")
n, bins, patches = plt.hist(D_filt.flatten()*1e4, 500, facecolor='tab:blue', alpha=0.75)
plt.xlabel(r'[$10^4$ s$^{-1}$]')
plt.axvline(x=0, ls = 'dashed',c='k')
plt.yscale('log')
plt.xlim((-5,5))
plt.ylim((1e2,1e5))

plt.tight_layout()

fig.savefig(dir_out+'filtered_fields_distribution.png', bbox_inches='tight')




## Plot filtered fields in the physical space
lim = 2 ## limit for imshow
lw = 5 ## contour width for the subdomains
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))

plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}$', fontsize = 34)
plt.imshow(TK_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)

plt.imshow(V_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(133)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
plt.imshow(D_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='green')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.tight_layout()


fig.savefig(dir_out+'filtered_fields_lim_'+str(lim)+'.png', bbox_inches='tight')
plt.close(fig)

## NEW figure including TAU Plot filtered fields in the physical space
lim = 2 ## limit for imshow
lw = 5 ## contour width for the subdomains
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(8*w,1.5*h))

plt.subplot(141)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\tau$', fontsize = 34)
plt.imshow(TauK_filt,origin='lower', vmin = -0.55, vmax = 0.55, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4,)
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(142)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta_{\tau}$', fontsize = 34)
plt.imshow(TK_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(143)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)

plt.imshow(V_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='gray')

plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.subplot(144)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
plt.imshow(D_filt*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
plt.plot(ys,xs, lw=lw)
plt.plot(ys_f1,xs_f1, lw=lw, c='tab:orange')
plt.plot(ys_f2,xs_f2, lw=lw, c='k')
# plt.plot(ys_f4,xs_f4, lw=4, c='green')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')
plt.xlim((0,D_filt.shape[1]))

plt.tight_layout()


fig.savefig(dir_out+'filtered_fields_lim_'+str(lim)+'_4fields.png', bbox_inches='tight')
plt.close(fig)



## plot physical fields as Fig 2.

## Plot filtered fields in the physical space
lim = 2## limit for imshow
lw = 4 ## contour width for the subdomains
plt.rcParams["text.usetex"] = True
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
fig = plt.figure(figsize=(6*w,1.5*h))


plt.subplot(131)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\zeta$', fontsize = 34)

plt.imshow(V*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')

# plt.plot(ys_f2,xs_f2, lw=lw, c='k')


plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.subplot(132)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 26)
plt.title(r'$\chi$', fontsize = 34)
plt.imshow(D*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')

# plt.plot(ys_f2,xs_f2, lw=lw, c='k')
plt.colorbar(shrink = 0.4, label = r'$[10^{4} \: \mathrm{s}^{-1}]$')
plt.xlabel(r'$\mathrm{x} \: [10 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [10 \mathrm{km}]$')

plt.tight_layout()


fig.savefig(dir_out+'zeta_div_fig2_filter.png', bbox_inches='tight')
plt.close(fig)



import matplotlib.offsetbox
from matplotlib.lines import Line2D
class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],
                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)



### plot the figure with the streamers and the scatter plots ##

x = np.arange(0,V_filt.shape[1],1) # tried to mimic your data boundaries
y = np.arange(0,V_filt.shape[0],1)
# x = np.arange(0,790,1) # tried to mimic your data boundaries
# y = np.arange(0,2080,1)
xx, yy = np.meshgrid(x,y)
m = np.all([yy>900,yy<(220/480)*xx+1202,yy<1500,xx>170,xx<650,yy>(210/480)*xx+825.625], axis = 0)

Vc = V_filt.copy()
Vc[~m]=np.nan
Dc = D_filt.copy()
Dc[~m]=np.nan
Tc = TK_filt.copy()
Tc[~m]=np.nan
# plt.imshow(V_box[600:1100,200:620]*1e4,origin='lower', vmin = -1, vmax = 1, cmap='RdBu_r')

lim = 2 ## param for imshow
fac = 2 ## param for the scatterplot scaling
plt.rcParams['font.size'] = '22'
# fig = plt.figure(figsize=(16,16))
fig, axs = plt.subplots(2, 2,figsize=(20,20))
plt.subplot(221)
plt.annotate("a", xy=(-0.15, 0.96), xycoords="axes fraction",weight='bold', size = 24)
plt.annotate(r'$\zeta_{\tau}$', xy=(0.36, 0.85), xycoords="axes fraction",weight='bold', size = 40)

plt.imshow(Tc*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
# plt.colorbar(shrink = 0.7)
plt.plot(ys_f2,xs_f2, c='gray',lw=2.5)
plt.gca().axis("off")

plt.ylim((895,1505))
plt.xlim((160,655))
ob = AnchoredHScaleBar(size=100, label=r"$\mathrm{1,000} \: \mathrm{km}$", loc=4, frameon=False,pad=0,sep=2, linekw=dict(color="k"),bbox_to_anchor=(1.15,.03), bbox_transform=plt.gca().transAxes,)
plt.gca().add_artist(ob)

plt.subplot(222)
plt.annotate("b", xy=(-0.15, 0.96), xycoords="axes fraction",weight='bold', size = 24)
plt.annotate(r'$\zeta$', xy=(0.37, 0.85), xycoords="axes fraction",weight='bold', size = 40)
pcm = plt.imshow(Vc*1e4,origin='lower', vmin = -lim, vmax = lim, cmap='RdBu_r')
# plt.colorbar(shrink = 0.7, label = r'[10$^4$ s$^{-1}$]')
plt.plot(ys_f2,xs_f2, c='gray',lw=2.5)
plt.gca().axis("off")
plt.ylim((895,1505))
plt.xlim((160,655))
fig.colorbar(pcm, ax=axs[0, :2], shrink=0.5, pad = 5, aspect = 35, location='bottom', label = r'$[10^4 \: \mathrm{s}^{-1}]$', anchor = (0.45,-0.4))

plt.subplot(223)

plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
# plt.scatter(M_vort[:,0,1],M_vort[:,1,1], s=10)
# plt.scatter(M_vort[:,0,0],M_vort[:,1,0], s=10, c='tab:orange')
plt.scatter(M_vort[:,0,2]/fac,M_vort[:,1,2], s=10 )
plt.scatter(M_vort[:,0,3]/fac,M_vort[:,1,3], s=10)
# plt.scatter(M_vort[:,0,4],M_vort[:,1,4], s=10, c='k')
# plt.plot(np.unique(M_vort[:,0,1]), np.poly1d(np.polyfit(M_vort[:,0,1], M_vort[:,1,1], 1))(np.unique(M_vort[:,0,1])), lw=2)
# plt.plot(np.unique(M_vort[:,0,0]), np.poly1d(np.polyfit(M_vort[:,0,0], M_vort[:,1,0], 1))(np.unique(M_vort[:,0,0])), color='tab:orange', lw=2)
plt.plot(np.unique(M_vort[:,0,2])/fac, np.poly1d(np.polyfit(M_vort[:,0,2]/fac, M_vort[:,1,2], 1))(np.unique(M_vort[:,0,2]/fac)), lw=2,label = 'streamer subdomain')
plt.plot(np.unique(M_vort[:,0,3])/fac, np.poly1d(np.polyfit(M_vort[:,0,3]/fac, M_vort[:,1,3], 1))(np.unique(M_vort[:,0,3]/fac)),  lw=2,label = 'full domain')

## add error bars
MM = M_vort[:,0,2].copy()/fac
list_std=np.linspace(-2.2,1.8,11)
list_std[-1]=1.7
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')

MM = M_vort[:,0,3].copy()/fac
list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')

plt.legend(frameon=False)
# plt.grid()
plt.ylim((-1.2,1.45))
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
# plt.ylabel(r'$\zeta_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
# plt.axvline(x=0,c='grey',linestyle='dashed',linewidth=2.2)
# plt.axhline(y=0,c='grey',linestyle='dashed',linewidth=2.2)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
plt.scatter(M_div[:,0,2]/fac,M_div[:,1,2],s=10)
plt.scatter(M_div[:,0,3]/fac,M_div[:,1,3], s=10)
plt.plot(np.unique(M_div[:,0,2])/fac, np.poly1d(np.polyfit(M_div[:,0,2]/fac, M_div[:,1,2], 1))(np.unique(M_div[:,0,2]/fac)), lw=2,label = 'streamer subdomain')
plt.plot(np.unique(M_div[:,0,3])/fac, np.poly1d(np.polyfit(M_div[:,0,3]/fac, M_div[:,1,3], 1))(np.unique(M_div[:,0,3]/fac)), lw=2,label = 'full domain')
## add error bars
## add error bars
MM = M_div[:,0,2].copy()/fac
list_std=np.linspace(-2,2,11)[1:]
list_std[-1]=1.7
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')

MM = M_div[:,0,3].copy()/fac
list_std=np.hstack((-1.7,np.linspace(-1.8,2.2,11)[2:]))
list_std[-1]=2.1
list_std[-2]=1.9
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')


plt.ylim((-1.2,1.2))
plt.legend(frameon=False)
plt.ylabel(r'$\chi \: [\: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.subplots_adjust( hspace = 0.25, wspace = 0.25)

# plt.tight_layout()
fig.savefig(dir_out+'scatter_fields_filtered_H_'+str(int(1e-3*g*alpha/N0**2))+'km_lim_'+str(lim)+'_b.png', bbox_inches='tight')
plt.close(fig)



# plt.subplot(313)
# plt.annotate("c", xy=(0, 0.7), xycoords="axes fraction",weight='bold', size = 24)
# plt.imshow(Dc*1e4,origin='lower', vmin = -1, vmax = 1, cmap='RdBu_r')
# plt.colorbar(shrink = 0.7)
# plt.plot(ys_f2,xs_f2, c='gray',lw=2)
# plt.gca().axis("off")
# plt.ylim((895,1505))
# plt.xlim((160,655))












### plot the 4 squares for zeta and zeta_tau##
fac = 2 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_vort[:,0,1],M_vort[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],M_vort[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], M_vort[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,1][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,1][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,0][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,0][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_vort[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_vort[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_vort[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_vort[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
# plt.ylabel(r'$div_{FT} \times 10^{-4}$ [s$^{-1}$]', fontsize=20)
plt.ylabel(r'$\zeta \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_vort[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_vort[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-1.2,1.45))
plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_vort.png', bbox_inches='tight')



### plot the 4 squares for div and zeta_tau##
fac = 2 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_div[:,0,1],M_div[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],M_div[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], M_div[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-2,2,11)[1:]
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = M_div[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,M_div[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.65, 0.8), xycoords="axes fraction",)
plt.scatter(MM,M_div[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, M_div[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\zeta_{\tau} \: [\: 2. \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.hstack((-1.7,np.linspace(-2,2,11)[2:]))
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]+M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,M_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]-M_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim((-1.2,1.45))
# plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_div.png', bbox_inches='tight')


### plot the 4 squares for div and tau##
fac = 1 # scaling for the x-axis
fig = plt.figure(figsize=(15,15))
plt.subplot(221)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,1].copy()/fac
# slope, intercept, r_value, p_value, std_err = stats.linregress(M_div[:,0,1],M_div[:,1,1])
slope, intercept, r_value, p_value, std_err = stats.linregress(MM[~np.isnan(MM)],T_div[:,1,1][~np.isnan(MM)])
slope
r_value**2

plt.title(r'Polar vortex')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,1], s=7)
plt.plot(np.unique(MM[~np.isnan(MM)]), np.poly1d(np.polyfit(MM[~np.isnan(MM)], T_div[:,1,1][~np.isnan(MM)], 1))(np.unique(MM[~np.isnan(MM)])), lw=2)
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [ \mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.4,0.65,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,1][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,1][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-0.4,0.6))

plt.subplot(222)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,0].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,0])
slope
r_value**2

plt.title(r'Lower left filament')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,0], s=7,c='tab:orange')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,0], 1))(np.unique(MM)), lw=2,c='tab:orange')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.98,0.6,11)
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,0][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,0][np.nanargmin(np.abs(lbin-MM))]/4],c='tab:orange')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-2,2.5))

plt.subplot(223)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,2].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,2])
slope
r_value**2
plt.title(r'Streamer')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,2], s=7,c='k')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,2], 1))(np.unique(MM)), lw=2,c='k')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.linspace(-0.7,0.7,11)[1:]
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,2][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,2][np.nanargmin(np.abs(lbin-MM))]/4],c='k')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim((-2,2.4))

plt.subplot(224)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 24)
MM = T_div[:,0,3].copy()/fac
slope, intercept, r_value, p_value, std_err = stats.linregress(MM,T_div[:,1,3])
slope
r_value**2

plt.title(r'Full domain')
plt.annotate('slope = '+str(np.round(slope,2))+'\nr$^2$ = '+str(np.round(r_value**2,2)),xy=(0.08, 0.8), xycoords="axes fraction",)
plt.scatter(MM,T_div[:,1,3], s=7,c='gray')
plt.plot(np.unique(MM), np.poly1d(np.polyfit(MM, T_div[:,1,3], 1))(np.unique(MM)), lw=2,c='gray')
plt.ylabel(r'$\chi \: [ \: 10^{4} \: \mathrm{s}^{-1}]$', fontsize=20)
plt.xlabel(r'$\tau \: [\mathrm{s}^{-1}]$', fontsize=20)

list_std=np.hstack((np.linspace(-1,1.3,11)[0:-4],0.61))
for lbin in list_std:
    plt.plot([MM[np.nanargmin(np.abs(lbin-MM))],MM[np.nanargmin(np.abs(lbin-MM))]],[T_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]+T_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4,T_div[:,1,3][np.nanargmin(np.abs(lbin-MM))]-T_div[:,2,3][np.nanargmin(np.abs(lbin-MM))]/4],c='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-0.7,1.5))
# plt.xlim((-2,2.4))

plt.tight_layout()

fig.savefig(dir_out+'scatter_subdomain_div_tau_bis.png', bbox_inches='tight')




## Plot the wind field as quiver plot
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = '22'
Int = 10
lim = 80
w, h = plt.figaspect(filt_u)
fig = plt.figure(figsize=(6*w,1.5*h))
plt.subplot(131)
plt.title(r'$\mathrm{Wind \: velocity}$')

U = filt_u
V = filt_v
M = np.sqrt(U**2+V**2)
M[M>lim]=lim
Int = 10
Q = plt.quiver( U[::Int,::Int], V[::Int,::Int],M[::Int,::Int],scale = 700, units = 'width', cmap = cmocean.cm.speed)
plt.quiverkey(Q, 0.24, 0.97, 50, r'50 $\mathrm{m.s}^{-1}$',labelpos ='E',coordinates ='figure')
plt.colorbar(shrink = 0.4,label = r'$\mathrm{Wind \: velocity} \; [\mathrm{m.s}^{-1}]$')
plt.xlim((0,U.shape[1]/intt))
plt.ylim((0,U.shape[0]/intt))
plt.xlabel(r'$\mathrm{x} \: [100 \mathrm{km}]$')
plt.ylabel(r'$\mathrm{y} \: [100 \mathrm{km}]$')
plt.tight_layout()

fig.savefig(dir_out+'quiver_uv_n0103_lim_'+str(lim)+'_weakbutterworth.png', bbox_inches='tight')


## checking Parseval
n = 1
wc = 250
## apply a butterworth low pass to u and v ##
sv, FILT = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)
spec_u_load, filt_u_load = jf.butterworh_lp(dx,dy,u_l,n,wl_c=wc)

Nj,Ni = filt_u_load.shape
wavnum1D,wavnum2D,kx,ky = jf.wvnumb_vector(dx,dy,Ni,Nj)

kr_load, E_u_load= jf.wv_spec(filt_u_load.copy(),dx,dy,detrend=None)


spec_fft = np.fft.fft2(filt_u_load)
spec_2D = (spec_fft*spec_fft.conj()).real*(dx*dy)/(Ni*Nj)
mod = (np.fft.fft2(filt_u_load).real**2+np.fft.fft2(filt_u_load).imag**2)*dx*dy/(Ni*Nj)
kr_load, E_u_jh= jf.calc_ispec(kx,ky,spec_2D)


np.sum(E_u_load)
np.sum(E_u_jh)


var_phys = np.sum(filt_u_load**2)/(Ni*Nj)
var_phys
var_spec = np.sum(spec_2D)/(Ni*Nj*dx*dy)
var_spec
var_spec = np.sum(mod)/(Ni*Nj*dx*dy)
var_spec
