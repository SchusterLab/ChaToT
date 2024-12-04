# Everything in here is co-opted from Elizabeth Panner's work on resonance fitting

import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt
import h5py
from functools import partial
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os

def removecable(f, z, tau, f1):
    """
    returns:
        z_no_cable:  z with the cable delay factor removed (guessing tau, relative to f1?)
    """
    z_no_cable = np.array(z)*np.exp(2j*np.pi*(np.array(f)-f1)*tau)
    return z_no_cable


def estpara(f, z, fr_0):
    """
    returns:
        f0_est:  The estimated center frequency for this resonance
        Qr_est:  The estimated total quality factor
        id_f0:   The estimated center frequency in index number space
        id_BW:   The 3dB bandwidth in index number space
    """

    edge_data_f = np.hstack((f[:int(len(f)/10)],f[-int(len(f)/10):]))
    edge_data_z = np.hstack((z[:int(len(f)/10)],z[-int(len(f)/10):]))

    realfit = np.polyfit(edge_data_f,edge_data_z.real,1)
    imagfit = np.polyfit(edge_data_f,edge_data_z.imag,1)
    zfinder = np.sqrt((z.real-(realfit[1]+f*realfit[0]))**2+(z.imag-(imagfit[1]+f*imagfit[0]))**2)
    edge_val = np.mean(zfinder)
    zfinder = (zfinder+np.append(edge_val,zfinder[:-1])+np.append(zfinder[1:],edge_val)+np.append([edge_val,edge_val],zfinder[:-2])+np.append(zfinder[2:],[edge_val,edge_val]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1)+np.append([1,1],zfinder[:-2])+np.append(zfinder[2:],[1,1]))/5
    #zfinder = (zfinder+np.append(1,zfinder[:-1])+np.append(zfinder[1:],1))/3
    left_trim = np.argmin(zfinder[f<fr_0])
    right_trim = np.argmin(abs(f-fr_0)) + np.argmin(zfinder[f>=fr_0])
    zfinder = zfinder - min(zfinder[left_trim], zfinder[right_trim])
    id_f0 = left_trim + np.argmax(zfinder[left_trim:right_trim+1])

    id_BW_left = left_trim + np.argmin(abs(abs(z[left_trim:id_f0]-z[id_f0])-abs(z[left_trim:id_f0]-np.mean(edge_data_z))))
    id_BW_right = id_f0 + np.argmin(abs(abs(z[id_f0:right_trim+1]-z[id_f0])-abs(z[id_f0:right_trim+1]-np.mean(edge_data_z))))

    if False:
        plt.figure(1)
        #plt.plot(f_f0_finder,abs(dz_f0_finder))
        plt.plot(f[left_trim:right_trim+1], abs(z[left_trim:right_trim+1]),'.')
        plt.plot(f, zfinder, '.')
        plt.plot(f[left_trim:right_trim+1], zfinder[left_trim:right_trim+1], '.')
        plt.axvline(x=fr_0)
        plt.axvline(x=f[id_f0], c="g")
        #plt.axvline(x=f[id_3db_left], color="red")
        #plt.axvline(x=f[id_3db_right], color="red")
        plt.axvline(x=f[id_BW_left], c="r")
        plt.axvline(x=f[id_BW_right], c="r")
        plt.axhline(y=0)
        #plt.axhline(y=z_3db, color="red")

        plt.figure(2)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag)
        plt.plot(realfit[1]+f*realfit[0],imagfit[1]+f*imagfit[0])
        plt.plot(z[id_f0].real,z[id_f0].imag,'o',c='g')
        plt.plot(z[id_BW_left].real,z[id_BW_left].imag,'o',c='r')
        plt.plot(z[id_BW_right].real,z[id_BW_right].imag,'o',c='r')
        plt.show()

    f0_est = f[id_f0]
    #id_BW = 2*np.mean([abs(id_f0-id_3db_left), abs(id_f0-id_3db_right)])
    id_BW = id_BW_right-id_BW_left
    #Qr_est = f0_est/(2*np.mean([abs(f[id_f0]-f[id_3db_left]), abs(f[id_f0]-f[id_3db_right])]))
    Qr_est = f0_est/(f[id_BW_right]-f[id_BW_left])

    return f0_est, Qr_est, id_f0, id_BW


def circle2(z):
    # == METHOD 2b ==
    # "leastsq with jacobian"
    x = z.real
    y = z.imag
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt(((x-xc)**2)+((y-yc)**2))

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri-Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_estimate = x_m, y_m
    center_2b, ier = opt.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b              # circle center
    Ri_2b = calc_R(*center_2b)            # distance of each data point from center_2b
    R_2b = Ri_2b.mean()                   # average Ri_2b, used as predicted radius
    residu_2b = sum((Ri_2b - R_2b)**2)    # residual?

    zc =  center_2b[0]+center_2b[1]*1j

    t = np.arange(0,2*np.pi,0.002)
    xcirc = center_2b[0]+R_2b*np.cos(t)
    ycirc = center_2b[1]+R_2b*np.sin(t)

    return residu_2b, zc, R_2b


def fitphase2(f,z,zc,fr,Qr,z_off):
    z_E3 = z*np.exp(-1j*np.angle(z_off))
    zc_E3 = zc*np.exp(-1j*np.angle(z_off))
    z_off_E3 = z_off*np.exp(-1j*np.angle(z_off))
    phi = np.pi + np.angle(zc_E3-z_off_E3)
    phi = np.angle(np.exp(1j*phi))
    z_no_phi = (z_E3-zc_E3)*np.exp(-1j*phi)

    def no_phi_eq(f_F,fr_F,Qr_F):
        return 4*Qr*(1-f_F/fr_F)/(1-4*Qr_F*Qr_F*((1-f_F/fr_F)**2))

    presults, pcov = opt.curve_fit(no_phi_eq,f,z_no_phi.imag/z_no_phi.real,p0=[fr,Qr])

    if False:
        plt.figure(1)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z_E3.real,z_E3.imag)
        plt.plot(zc_E3.real,zc_E3.imag,'*')
        plt.plot(z_off_E3.real,z_off_E3.imag,'o',fillstyle='none',markersize=15)
        plt.plot(z_no_phi.real,z_no_phi.imag)

        plt.figure(2)
        plt.plot(f,z_no_phi.imag/z_no_phi.real)
        #plt.plot(f,no_phi_eq(f,fr,Qr))
        plt.plot(f,no_phi_eq(f,presults[0],presults[1]))

        plt.show()

    return presults[1], presults[0], phi

# Resonance fitting function
def roughfit(f, z, fr_0, fit_res_obj=None, plot=False):

    # what does this do?
    edge_data_f = np.hstack((f[:int(len(f)/100)],f[-int(len(f)/100):]))
    edge_data_z = np.hstack((z[:int(len(f)/100)],z[-int(len(f)/100):]))

    # gets index of resonance frequency)
    id_f0 = np.argmin(abs(f-fr_0))

    tau_fit = np.polyfit(edge_data_f-f[id_f0],np.angle(edge_data_z/z[id_f0]),1)
    tau_1 = tau_fit[0]/(-2*np.pi)

    if plot:
        plt.plot(edge_data_f-f[id_f0],np.angle(edge_data_z/z[id_f0]),'.')
        plt.plot(edge_data_f-f[id_f0],tau_fit[0]*(edge_data_f-f[id_f0])+tau_fit[1])
        plt.show()

    tau_fit = np.polyfit(edge_data_f-f[id_f0],np.log(abs(edge_data_z/z[id_f0])),1)
    Imtau_1 = tau_fit[0]/(2*np.pi)

    # remove cable term
    z1 = removecable(f, z, tau_1+1j*Imtau_1, fr_0)

    if plot:
        #print tau_1
        plt.figure(1)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag)
        plt.plot(z1.real,z1.imag)
        plt.figure(2)
        plt.plot(f,abs(z))
        plt.plot(f,abs(z1))
        #plt.show()

    # estimate f0 (pretty good), Q (very rough)
    f0_est, Qr_est, id_f0, id_BW = estpara(f,z1,fr_0)

    ## Save the estimates to our class instance
    if (fit_res_obj is not None):
        fit_res_obj.f0_est = f0_est
        fit_res_obj.Qr_est = Qr_est
        fit_res_obj.id_f0  = id_f0
        fit_res_obj.id_BW  = id_BW

    # fit circle using trimmed data points
    id1 = max(id_f0-int(0.5*id_BW), 0)
    id2 = min(id_f0+int(0.5*id_BW), len(f))
    if len(range(id1, id2)) < 3:
        id1 = 0
        id2 = len(f)

    residue, zc, r = circle2(z1[id1:id2])

    # rotation and traslation to center
    z1b = z1*np.exp(-1j*np.angle(zc, deg=False))
    z2 = (zc-z1)*np.exp(-1j*np.angle(zc, deg=False))

    if plot:
        plt.figure(2)
        plt.axvline(x=0, color='gray')
        plt.axhline(y=0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(z.real,z.imag,'o', color=None, label='z')
        plt.plot(z1.real,z1.imag,'o', color=None, label='z1')
        #plt.plot(z1b.real,z1b.imag,'o', color=None, label='z1b')
        #plt.plot(z2.real,z2.imag,'o', color=None, label='z2')
        #plt.plot(z2[np.argmin(abs(f-f0_est))].real,z2[np.argmin(abs(f-f0_est))].imag,'o', color='orange', label='f0 est')
        plt.legend()
        plt.show()

    # fit phase
    z_off = np.mean(np.hstack((z1[:int(len(f)/10)],z1[-int(len(f)/10):])))
    Q, f0, phi = fitphase2(f,z1,zc,f0_est,Qr_est,z_off)

    Qc = abs(z_off)*Q/(2*r)

    if plot:
        z_rough = z_off*np.exp(-2j*np.pi*(f-f0)*(tau_1+1j*Imtau_1))*(1-((Q/Qc)*np.exp(1j*phi))/(1+2j*Q*((f-f0)/f0)))
        plt.figure(1)
        plt.plot(z.real,z.imag)
        plt.plot(z_rough.real,z_rough.imag)
        plt.plot(z_off.real,z_off.imag,'*')
        plt.figure(2)
        plt.plot(f,z.real)
        plt.plot(f,z.imag)
        plt.plot(f,z_rough.real)
        plt.plot(f,z_rough.imag)
        plt.show()

    ## Create a dictionary for the output
    result = {  "f0"    : f0, 
                "Q"     : Q,
                "phi"   : phi,
                "zOff"  : z_off,
                "Qc"    : Qc,
                "tau1"  : tau_1,
                "Imtau1": Imtau_1}

    ## Write the rough fit result to the fit result class instance
    if (fit_res_obj is not None):
        fit_res_obj.rough_result["f0"]     = result["f0"]
        fit_res_obj.rough_result["Q"]      = result["Q"]
        fit_res_obj.rough_result["phi"]    = result["phi"]
        fit_res_obj.rough_result["zOff"]   = result["zOff"]
        fit_res_obj.rough_result["Qc"]     = result["Qc"]
        fit_res_obj.rough_result["tau1"]   = result["tau1"]
        fit_res_obj.rough_result["Imtau1"] = result["Imtau1"]

    return result


# S21 transmission function
def S21(f, fr, Qr, Qc_hat_mag, a, phi, tau):
    """A semi-obvious form of Gao's S21 function. e^(2j*pi*fr*tau) is incorporated into a."""
    S21 = a*np.exp(-2j*np.pi*(f-fr)*tau)*(1-(((Qr/Qc_hat_mag)*np.exp(1j*phi))/(1+(2j*Qr*(f-fr)/fr))))
    return S21