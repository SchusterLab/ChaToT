from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

def hist(data=None, plot=True, ran=1.0): # Shannon Harvey
    
    ig = data[0]
    qg = data[1]
    ie = data[2]
    qe = data[3]

    numbins = 200
    
    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    if plot==True:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        fig.tight_layout()

        axs[0].scatter(ig, qg, label='g', color='b', marker='*')
        axs[0].scatter(ie, qe, label='e', color='r', marker='*')
        axs[0].scatter(xg, yg, color='k', marker='o')
        axs[0].scatter(xe, ye, color='k', marker='o')
        axs[0].set_xlabel('I (a.u.)')
        axs[0].set_ylabel('Q (a.u.)')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated')
        axs[0].axis('equal')
    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    """Rotate the IQ data"""
    ig_new = ig*np.cos(theta) - qg*np.sin(theta)
    qg_new = ig*np.sin(theta) + qg*np.cos(theta) 
    ie_new = ie*np.cos(theta) - qe*np.sin(theta)
    qe_new = ie*np.sin(theta) + qe*np.cos(theta)
    
    """New means of each blob"""
    xg, yg = np.median(ig_new), np.median(qg_new)
    xe, ye = np.median(ie_new), np.median(qe_new)
    
    #print(xg, xe)
    
    xlims = [xg-ran, xg+ran]
    ylims = [yg-ran, yg+ran]

    if plot==True:
        axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
        axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
        axs[1].scatter(xg, yg, color='k', marker='o')
        axs[1].scatter(xe, ye, color='k', marker='o')    
        axs[1].set_xlabel('I (a.u.)')
        axs[1].legend(loc='lower right')
        axs[1].set_title('Rotated')
        axs[1].axis('equal')

        """X and Y ranges for histogram"""
        
        ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        axs[2].set_xlabel('I(a.u.)')       
        
    else:        
        ng, binsg = np.histogram(ig_new, bins=numbins, range = xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range = xlims)

    """Compute the fidelity using overlap of the histograms"""
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    threshold=binsg[tind]
    fid = contrast[tind]
    axs[2].set_title(f"Fidelity = {fid*100:.2f}%")

    return fig


class single_shot_ground(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len)

        self.add_loop(name="iter_loop", count=cfg.expt.n_count)

        self.add_readoutconfig(ch=cfg.soc.ro_ch, name='ro', freq=cfg.expt.res_freq, gen_ch=cfg.soc.res_gen_ch)

        self.add_pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", ro_ch=cfg.soc.ro_ch, 
                       style="const", 
                       length=cfg.expt.res_pulse_len,
                       freq=cfg.expt.res_freq, 
                       phase=cfg.expt.res_phase,
                       gain=cfg.expt.res_gain, 
                      )

    def _body(self, cfg):
        cfg = AttrDict(cfg)
        self.send_readoutconfig(ch=cfg.soc.ro_ch, name='ro', t=0)
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.pulses.pi_const.length+0.01) # maybe remove the delay?
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.pulses.pi_const.length+0.01, ddr4=True)

class single_shot_excited(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len/2)

        self.add_loop(name="iter_loop", count=cfg.expt.n_count)

        self.add_readoutconfig(ch=cfg.soc.ro_ch, name='ro', freq=cfg.expt.res_freq, gen_ch=cfg.soc.res_gen_ch)

        self.add_pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", ro_ch=cfg.soc.ro_ch, 
                       style="const", 
                       length=cfg.expt.res_pulse_len,
                       freq=cfg.expt.res_freq, 
                       phase=cfg.expt.res_phase,
                       gain=cfg.expt.res_gain, 
                      )
        
        self.add_gauss(ch=cfg.soc.qubit_gen_ch, name="ramp", sigma=cfg.pulses.pi_gaus.length/10, length=cfg.pulses.pi_gaus.length, even_length=True)
        self.add_pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
                       style="arb", 
                       envelope="ramp", 
                       freq=cfg.pulses.pi_gaus.freq, 
                       phase=cfg.pulses.pi_gaus.phase,
                       gain=cfg.pulses.pi_gaus.gain
                      )

    def _body(self, cfg):
        cfg = AttrDict(cfg)
        self.send_readoutconfig(ch=cfg.soc.ro_ch, name='ro', t=0)
        self.pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", t=0)
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.pulses.pi_const.length+0.01)
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.pulses.pi_const.length, ddr4=True)


class single_shot_filtering(Experiment):
    def __init__(self, path='', prefix='single_shot_filtering', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        return

    def acquire(self, progress=False):
        super().acquire()
        
        prog_ground = single_shot_ground(soccfg=self.soccfg, reps=1, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        iq_list_ground = []
        for n in tqdm(range(self.cfg.expt.n_count)):
            iq_list_ground.append(prog_ground.acquire_decimated(self.soc, soft_avgs=1, progress=False)[0])
        
        prog_excited = single_shot_excited(soccfg=self.soccfg, reps=1, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        iq_list_excited = []
        for n in tqdm(range(self.cfg.expt.n_count)):
            iq_list_excited.append(prog_excited.acquire_decimated(self.soc, soft_avgs=1, progress=False)[0])

        t = prog_ground.get_time_axis(ro_index=0) # this may cause problems
        Ig_filtered = np.array(iq_list_ground)[:,:,0] * self.cfg.expt.filter(len(t))
        Qg_filtered = np.array(iq_list_ground)[:,:,0] * self.cfg.expt.filter(len(t))
        Ie_filtered = np.array(iq_list_excited)[:,:,0] * self.cfg.expt.filter(len(t))
        Qe_filtered = np.array(iq_list_excited)[:,:,0] * self.cfg.expt.filter(len(t))

        self.data = {"Ig": Ig_filtered, "Qg": Qg_filtered, "Ie": Ie_filtered, "Qe": Qe_filtered}
        return self.data

    def display(self, save=True):
        i_g = self.data["Ig"]
        q_g = self.data["Qg"]
        i_e = self.data["Ie"]
        q_e = self.data["Qe"]
        fig = hist([i_g, q_g, i_e, q_e], ran=5)
        
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return












