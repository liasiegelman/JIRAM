## loading packages - Feb, 24, 2023 - lsiegelman ##
import pandas as pd
from matplotlib import colors as cols
from xhistogram.xarray import histogram
import xarray as xr
import gcsfs
import numpy as np
import matplotlib
import xarray
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
sys.path.append('/Users/'+user+'/Dropbox/Jovian_dynamics/programs/')
import Powerspec_a as ps
import jupiter_functions as jf
import twodspec  as spec
from numpy import ma
import Powerspec_a as ps
import math
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Box2DKernel
from datetime import date

## make repertory for today's plots
today = date.today()
d1 = today.strftime("%Y%m%d")
dir_out = '/Users/'+user+'/Dropbox/Jovian_dynamics/plot/'+d1+'/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)


## defining parameters for Jupiter atmosphere ##
g = 29.8
f0 = 3.5e-4
N0 = 4e-3
H0 = 9000
alpha = H0*N0**2/g 
N02=N0**2
N0of0=N0/f0

## resolution of the data, here 10km/pixel
dx = 1e4
dy = 1e4

## upload infrared image and make it doubly periodic ##
dir = '/Users/'+user+'/Dropbox/Jovian_dynamics/data/feature_tracking/mosaic/n0103/'
u_load = np.load(dir+'u_n0103.npy')
v_load = np.load(dir+'v_n0103.npy')
im = np.load(dir+'im_n02.npy')
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


## divergence FT as in Nat. Physics
n = 2
wc = 500 # this is the Nat. Physics paper 

## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)

PSI_b, VORT_b, DIV_b, SIG_N_b, SIG_S_b = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)


## retrieve zeta tau like in Nat. Phys. 
PSI_tau, VORT_tau, U_tau, V_tau, SIG_N_tau, SIG_S_tau = jf.psi_from_tau(dx,dy,tau,g,alpha,N0,detrend=None) ## g*alpha/N0=37.25 is equivalent to H_0 * N

spec_psi_hp, psi_hp = jf.butterworh_hp(dx,dy,PSI_tau[:,:,1].real,n,wl_c=1600)

## retrieve psi SQG first from feature tracking and SQG ##
wl_c = 1/6e-4
n = 1
wc = 250 # 200 for n0204

PSI_comb , U_comb, V_comb, VORT_comb, DIV_comb, SIG_N_comb, SIG_S_comb = jf.FT_TAU(dx,dy,u,v,tau,g,alpha,N0,wl_c=wl_c,detrend=None)
STRAIN_comb = np.sqrt(SIG_N_comb[:,:,1].real**2+SIG_S_comb[:,:,1].real**2)

# ## like in the SI of the Nat Phys paper - for figure of physical fields
n = 1
wc = 250
## apply a butterworth low pass to u and v ##
spec_u, filt_u = jf.butterworh_lp(dx,dy,u,n,wl_c=wc)
spec_v, filt_v = jf.butterworh_lp(dx,dy,v,n,wl_c=wc)
_, _, DIV_raw, _, _ = jf.psi_from_u(dx,dy,filt_u,filt_v, detrend=None)

wl_low = 50
spec_thp, V_filt = jf.cut_lp(dx, dy, VORT_comb[im.shape[0]:, :im.shape[1], 1].real, (1/wl_low)*1e-3)
spec_thp, S_filt = jf.cut_lp(dx, dy, STRAIN_comb[im.shape[0]:, :im.shape[1]], (1/wl_low)*1e-3)

# %%

# Figure of physical fields 
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'
plt.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(7*w,2.5*h))
plt.subplot(141)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\tau$')
plt.imshow(Tau, origin = 'lower', cmap = 'bone')
plt.colorbar(label = r'$\tau$ ', shrink = 0.2)

plt.subplot(142)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\sigma$')
plt.imshow(S_filt*1e4, vmin = 0, vmax = 4, origin = 'lower', cmap = 'Blues')
plt.colorbar(label = r'$\sigma$ [10$^{-4}$ s$^{-1}$]', shrink = 0.2)

plt.subplot(143)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction", weight='bold', size=30)
plt.title(r'$\zeta$')
plt.imshow(V_filt*1e4, vmin = -3, vmax = 3, origin = 'lower', cmap = 'RdBu_r')
plt.colorbar(label = r'$\zeta$ [10$^{-4}$ s$^{-1}$]', shrink = 0.2)

plt.subplot(144)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size=30)
plt.title(r'$\chi$')
plt.imshow(DIV_raw[im.shape[0]:,:im.shape[1],1].real*1e4, vmin = -2, vmax = 2, origin = 'lower', cmap = 'RdBu_r')
plt.colorbar(label = r'$\chi$ [10$^{-4}$ s$^{-1}$]', shrink = 0.2)

plt.tight_layout()
# plt.show()

fig.savefig(dir_out+'fields.png', bbox_inches='tight', dpi=300)
plt.close(fig)



fig = plt.figure(figsize=(4*w,0.75*h))
plt.subplot(141)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\tau$')
plt.imshow(Tau, origin = 'lower', cmap = 'bone')
plt.colorbar(label = r'$\tau$ ', shrink = 0.8)

plt.subplot(142)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\psi_{\tau_{hp}}$')
plt.imshow(psi_hp[im.shape[0]:, :im.shape[1]], origin='lower', cmap='bone')
plt.colorbar(label=r'$\psi_{\tau_{hp}}$', shrink=0.8)

plt.subplot(143)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\psi_{\tau}$')
plt.imshow(PSI_tau[im.shape[0]:, :im.shape[1],1].real, origin='lower', cmap='bone')
plt.colorbar(label=r'$\psi_{\tau}$', shrink=0.8)

plt.subplot(144)
plt.annotate("d", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
plt.title(r'$\psi_{comb}$')
plt.imshow(PSI_comb[im.shape[0]:, :im.shape[1],1].real, origin='lower', cmap='bone')
plt.colorbar(label=r'$\psi_{comb}$', shrink=0.8)



# plt.show()
plt.tight_layout()
fig.savefig(dir_out+'psi_physical_b.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# compute plot in strain vorticity space
V = VORT_comb[im.shape[0]:,:im.shape[1],1].real
S = STRAIN_comb[im.shape[0]:,:im.shape[1]]
D = DIV_b[im.shape[0]:,:im.shape[1],1].real
T = Tau
P = PSI_comb[im.shape[0]:,:im.shape[1],1].real



YC = np.arange(0, V.shape[0]*dy, dy)
XC = np.arange(0, V.shape[1]*dx, dx)

## scatter plot ##
strain01_0 = xr.DataArray(S, coords=[YC, XC], dims=['YC', 'XC'])
vort01_0 = xr.DataArray(V, coords=[YC, XC], dims=['YC', 'XC'])

bband01 = np.linspace(-3, 3, 240)
cband01 = np.linspace(0, 2, 240)

binbox01 = (bband01[1]-bband01[0])*(cband01[1]-cband01[0])

F = f0
b_01_0 = (vort01_0/F).rename('vort1_0')
c_01_0 = (strain01_0/F).rename('strain1_0')

b_01_0 = b_01_0.chunk({'XC': V.shape[1], 'YC': V.shape[0]})
c_01_0 = c_01_0.chunk({'XC': V.shape[1], 'YC': V.shape[0]})

print('compute bi hist')
hab01_00 = histogram(b_01_0, c_01_0, dim=['XC', 'YC'], bins=[bband01, cband01])
hab01_00.load()
print('bi hist complete')

print('compute histogram tau')
t = xr.DataArray(T, coords=[YC, XC], dims=['YC', 'XC'])
aa_01_100 = t.rename('T')
hab01_tau = histogram(b_01_0, c_01_0, weights=aa_01_100, dim=[
                      'XC', 'YC'], bins=[bband01, cband01])
hab01_tau.load()
print('histogram tau complete')

print('compute histogram div_ft')
div_ft = xr.DataArray(D/F, coords=[YC, XC], dims=['YC', 'XC'])
ab_01_100 = div_ft.rename('div_ft')

h_ab01_100 = histogram(b_01_0, c_01_0, weights=ab_01_100, dim=[
                       'XC', 'YC'], bins=[bband01, cband01])
h_ab01_100.load()
print('histogram div complete')

print('compute histogram psi_comb')
psi = xr.DataArray(P, coords=[YC, XC], dims=['YC', 'XC'])
h_psi0 = psi.rename('psi')

h_psi = histogram(b_01_0, c_01_0, weights=h_psi0, dim=[
                  'XC', 'YC'], bins=[bband01, cband01])
h_psi.load()
print('histogram psi_comb complete')




## compute density for chi FT like balwada 2021

q = np.array(h_psi)
m = np.array(h_ab01_100)
n = np.array(hab01_00)
o = np.array(hab01_tau)
pmax = n.max()
p_list = np.arange(0, pmax+1)

refer1 = hab01_00
pgrid1 = np.linspace(0,refer1.max(),len(p_list))

verProb1 = []
chi_plus = []
chi_moins = []
chi_tau_plus = []
chi_tau_moins = []
chi_psi_plus = []
chi_psi_moins = []

for i in range(len(p_list)):
    mask_pr1 = xr.where(refer1 < pgrid1[i], 0, 1)
    verProb1.append((refer1*mask_pr1).sum())
    chi_plus.append((m[m>0]*np.array(mask_pr1)[m>0]).sum())
    chi_moins.append((m[m<0]*np.array(mask_pr1)[m<0]).sum())
    chi_tau_plus.append((m[m>0]*o[m>0]*np.array(mask_pr1)[m>0]).sum())
    chi_tau_moins.append((m[m<0]*o[m<0]*np.array(mask_pr1)[m<0]).sum())
    chi_psi_plus.append((m[m>0]*q[m>0]*np.array(mask_pr1)[m>0]).sum())
    chi_psi_moins.append((m[m<0]*q[m<0]*np.array(mask_pr1)[m<0]).sum())



print('Start plotting')
plt.rcParams['xtick.major.pad']='12'
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = '22'
lim = 2.5
w, h = plt.figaspect(u_per)
plt.rcParams['font.size'] = '22'

filter01 = xr.where((hab01_00.rename('')).T>1e-5, 1, np.NaN)
# filter01 = xr.where((binbox01*XC.shape[0]*YC.shape[0])*(hab01_00.rename('')).T>1e-5, 1, np.NaN)

# fig = plt.figure(figsize=(25,16)) #size for 2x2 panels
fig = plt.figure(figsize=(8,14)) #size for 1 x 3 panels
plt.subplot(311)
plt.annotate("a", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)

girbNums01 = len(YC)*len(XC)
# test = (1/(binbox01*girbNums01)*(hab01_00.rename('')).T)
test = (hab01_00.rename('')).T
test = xr.where(test < 1e-5, np.nan, test)
test.plot(vmax=1e3, norm=cols.SymLogNorm(1e-4), cmap='Reds')


# # test = ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*(hab01_00.rename('')).T)  ### normalize result from xhistogram such that the sum*bin area = 1
# # test = xr.where(test<1e-5,np.nan,test)          ### here I set negligible probability to be nan so that the background of the plot will be white instead of slightly red
# # # xr.plot.contourf(test, vmax=5, norm=cols.SymLogNorm(1e-4),levels=10, cmap='Reds')
# # xr.plot.contourf(test,vmax=1e2,levels=[1e-3,1e-2,1e-1,0.7, 1.5], cmap ='Reds')


# plt.contour(bband01[:-1], cband01[:-1], ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*hab01_00.T), levels=[1e-2],colors ='k',alpha=.3)
# plt.contour(bband01[:-1], cband01[:-1], ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*hab01_00.T), levels=[1e-1],colors ='k',alpha=.3)
# plt.contour(bband01[:-1], cband01[:-1], ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*hab01_00.T), levels=[0.7],colors ='k',alpha=.3)
# plt.contour(bband01[:-1], cband01[:-1], ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*hab01_00.T), levels=[1.5],colors ='k',alpha=.3)

plt.contour(bband01[:-1], cband01[:-1], (hab01_00.T), levels=[10],colors ='k',alpha=.3)
plt.contour(bband01[:-1], cband01[:-1], (hab01_00.T), levels=[37.75],colors ='k',alpha=.3)
plt.contour(bband01[:-1], cband01[:-1], (hab01_00.T), levels=[75.5],colors ='k',alpha=.3)
plt.contour(bband01[:-1], cband01[:-1], (hab01_00.T), levels=[252],colors ='k',alpha=.3)
plt.contour(bband01[:-1], cband01[:-1], (hab01_00.T), levels=[500],colors ='k',alpha=.3)

plt.plot(np.linspace(0,-3,10),np.linspace(0,3,10),'k:',alpha=.5)
plt.plot(np.linspace(0,3),np.linspace(0,3),'k:',alpha=.5)
plt.xlabel(r'$\zeta/f_0$', fontsize=20)
plt.ylabel(r'$\sigma/f_0$', fontsize=20)
plt.title(r'$\mathrm{vorticity–strain} \: \mathrm{JDF}$', fontsize=30)
plt.text(-.1,1.75,r'$\mathrm{SD}$')
plt.text(1.8,.8,r'$\mathrm{CVD}$')
plt.text(-2.4,.8,r'$\mathrm{AVD}$')
plt.rc('grid', color='black', alpha=.3)
plt.grid()
plt.ylim((0, 2))
plt.xlim((-lim, lim))


plt.subplot(312)
plt.annotate("b", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
ii = (1/(len(XC)*len(YC)))*(hab01_tau.rename('').T)*filter01
# ii= ((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*(hab01_tau.rename('')).T)*filter01
# xr.plot.contourf(ii,cmap='RdBu_r')
xr.plot.contourf(ii, vmin=-3.5e-5, vmax=3.5e-5, levels=[-3e-5, -2*1e-5, -1e-5, -0.5e-5, -0.1*1e-5, -0.05*1e-5, 0, 0.05*1e-5, 0.1*1e-5,  0.5e-5, 1e-5, 2*1e-5, 3e-5], cmap='RdBu_r')
# xr.plot.contourf(ii, vmin = -0.55, vmax=0.55, levels=[-0.15, -0.1, -0.05,-0.01,-0.005, 0,0.005,0.01, 0.05, 0.1, 0.15], cmap='RdBu_r')
plt.plot(np.linspace(0,-7,10),np.linspace(0,7,10),'k--')
plt.plot(np.linspace(0,7),np.linspace(0,7),'k--')
plt.xlabel(r'$\zeta/f_0$', fontsize=20)
plt.ylabel(r'$\sigma/f_0$', fontsize=20)
plt.title(r'$\tau$', fontsize=30)
plt.ylim((0,2))
plt.xlim((-lim, lim))
plt.rc('grid', color='black', alpha=.3)
plt.grid()


plt.subplot(313)
plt.annotate("c", xy=(0, 1.02), xycoords="axes fraction",weight='bold', size = 30)
# cc = (1/binbox01)*(h_ab01_100.rename('').T)*filter01
cc = (1/(len(XC)*len(YC)))*(h_ab01_100.rename('').T)*filter01
# cc=((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*(h_ab01_100.rename('')).T)*filter01
xr.plot.contourf(cc, vmin=-4e-6, vmax=1.8e-5,levels=[-2.25e-6,-2e-6, -1e-6, -0.5e-6, -0.25e-6, -0.1e-6, 0, 0.1e-5, 0.25e-5, 0.5e-5, 0.75e-5, 1e-5, 1.25e-5], cmap='RdBu_r')
# xr.plot.contourf(cc, cmap='RdBu_r')
# xr.plot.contourf(cc,vmin = -1e-2, vmax=5e-2,  levels=[-0.01,-0.005,-0.0025,-0.00125,0,0.01,0.02,0.03,0.04,0.05], cmap='RdBu_r')
plt.plot(np.linspace(0,-7,10),np.linspace(0,7,10),'k--')
plt.plot(np.linspace(0,7),np.linspace(0,7),'k--')
plt.xlabel(r'$\zeta/f_0$', fontsize=20)
plt.ylabel(r'$\sigma/f_0$', fontsize=20)
plt.title(r'$\chi/f_0$', fontsize=30)
plt.ylim((0,2))
plt.xlim((-lim, lim))
plt.rc('grid', color='black', alpha=.3)
plt.grid()

plt.tight_layout()

plt.show()


fig.savefig(dir_out+'scatter_div_new.png', bbox_inches='tight', dpi=300)
plt.close(fig)


## plot xi_sqg(zeta, sigma)
filter01 = xr.where((binbox01*XC.shape[0]*YC.shape[0])*(hab01_00.rename('')).T>1e-5, 1, np.NaN)

fig = plt.figure(figsize=(16,8)) #size for 1 x 3 panels
dd=((1**2)/(binbox01*XC.shape[0]*YC.shape[0])*(hbb01.rename('')).T)*filter01
xr.plot.contourf(dd, vmin = -2e-3, vmax=2e-3, levels=5, cmap='RdBu_r')
plt.plot(np.linspace(0,-7,10),np.linspace(0,7,10),'k--')
plt.plot(np.linspace(0,7),np.linspace(0,7),'k--')
plt.xlabel(r'$\zeta/f_0$', fontsize=20)
plt.ylabel(r'$\sigma/f_0$', fontsize=20)
plt.title(r'$\chi_{sqg}/f_0$', fontsize=30)
plt.ylim((0,2))
plt.xlim((-3, 3))
plt.rc('grid', color='black', alpha=.3)
plt.grid()
plt.tight_layout()
plt.show()

fig.savefig(dir_out+'scatter_xi_sqg.png', bbox_inches='tight')
plt.close(fig)


fig, ax1 = plt.subplots(figsize=(10,7)) 
ax1.semilogx((refer1.max().values)*(1/pgrid1), chi_plus/np.max(chi_plus), label=r'$\chi^{+}/\overline{\chi^{+}}$')
ax1.semilogx((refer1.max().values)*(1/pgrid1), np.array(chi_moins)/np.min(chi_moins), label=r'$\chi^{-}/\overline{\chi^{-}}$')
# ax1.semilogx((refer1.max().values)*(1/pgrid1), chi_psi_plus/ np.max(chi_psi_plus), label=r'$ \chi^{+} . \psi / \overline{\chi^{+} . \psi} $')
# ax1.semilogx((refer1.max().values)*(1/pgrid1), chi_psi_moins/ np.max(chi_psi_moins), label=r'$ \chi^{-} . \psi / \overline{\chi^{-} . \psi} $')
# ax1.semilogx((refer1.max().values)*(1/pgrid1), chi_tau_plus/ np.max(chi_tau_plus), label=r'$ \chi^{+} . \tau / \overline{\chi^{+} . \tau} $')
# ax1.semilogx((refer1.max().values)*(1/pgrid1), chi_tau_moins/ np.max(chi_tau_moins), label=r'$ \chi^{-} . \tau / \overline{\chi^{-} . \tau} $', alpha = 0.7)

ax1.semilogx((refer1.max().values)*(1/pgrid1), np.array(verProb1)/refer1.values.sum(), 'k', linestyle='dotted', markersize=3, label=r'$\mathrm{Area}$')
ax1.set_xlabel(r'$d_{max}/d$')
ax1.set_ylabel(r'$\mathrm{Fraction}$')
ax1.legend()
# ax1.rc('grid', color='black', alpha=.3)
ax1.grid()
ax1.set_xlim((1, 755))

ax2 = ax1.twiny() 
ax2.set_xlabel('$\mathrm{Wavelength} \; \mathrm{[km]}$')
# ax2.set_xscale('log')
ax2.set_xticks([520, 230, 150, 100, 75])
ax2.invert_xaxis()
# ax2.set_xlim((1, 755))


# ax2.set_ylabel('Horizontal Area Fraction')
# ax2.plot((refer1.max().values)*(1/pgrid1),np.array(verProb1)/refer1.values.sum(),'k',linestyle='dotted', markersize = 3)
# # ax2.plot((refer1.max().values)*(1/pgrid1),mask_pr1.sum(),'k',linestyle='dotted', markersize = 3)
# # ax2.set_ylim((0,1.067))
# # ax2.set_xlim((0,1.1e3))
# ax2.set_yticks([0., 0.2, 0.4, 0.6, .8, 1])

plt.tight_layout()
plt.show()

fig.savefig(dir_out+'density_chi_psi_test.png', bbox_inches='tight', dpi=300)
plt.close(fig)
# %%

