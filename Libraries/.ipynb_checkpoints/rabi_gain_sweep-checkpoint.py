from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

class rabi_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len)

        self.add_loop(name="freq_loop", count=cfg.expt.freq_steps)
        self.add_loop(name="gain_loop", count=cfg.expt.gain_steps)

        self.add_readoutconfig(ch=cfg.soc.ro_ch, name='ro', freq=cfg.expt.res_freq, gen_ch=cfg.soc.res_gen_ch)

        self.add_pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", ro_ch=cfg.soc.ro_ch, 
                       style="const", 
                       length=cfg.expt.res_pulse_len,
                       freq=cfg.expt.res_freq, 
                       phase=cfg.expt.res_phase,
                       gain=cfg.expt.res_gain, 
                      )

        # constant qubit pulse
        # self.add_pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
        #                style="const", 
        #                length=cfg.expt.qubit_pulse_len,
        #                freq=cfg.expt.qubit_freq, 
        #                phase=cfg.expt.qubit_phase,
        #                gain=cfg.expt.qubit_gain, 
        #               )

        # gaussian qubit pulse
        self.add_gauss(ch=cfg.soc.qubit_gen_ch, name="ramp", sigma=cfg.expt.qubit_pulse_len/10, length=cfg.expt.qubit_pulse_len, even_length=True)
        self.add_pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
                       style="arb", 
                       envelope="ramp", 
                       freq=cfg.expt.qubit_freq, 
                       phase=cfg.expt.qubit_phase,
                       gain=cfg.expt.qubit_gain, 
                      )

    def _body(self, cfg):
        cfg = AttrDict(cfg)
        
        self.send_readoutconfig(ch=cfg.soc.ro_ch, name='ro', t=0)
        self.pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", t=0)
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.expt.qubit_pulse_len+0.01)
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.expt.qubit_pulse_len, ddr4=False)


class rabi_gain_sweep(Experiment):
    def __init__(self, path='', prefix='rabi_gain_sweep', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        return

    def acquire(self, progress=False):
        super().acquire()
        prog = rabi_pulse(soccfg=self.soccfg, reps=self.cfg.expt.n_avg, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        fs = np.linspace(self.cfg.expt.min_freq, self.cfg.expt.max_freq, self.cfg.expt.freq_steps)
        gs = np.linspace(self.cfg.expt.min_gain, self.cfg.expt.max_gain, self.cfg.expt.gain_steps) # relaxation time should be somewhere else in the cfg
        iq_list = prog.acquire(self.soc)
        self.data = {"I": iq_list[0][0][:,:,0], "Q": iq_list[0][0][:,:,1],"fs": fs, "gs": gs}
        return self.data

    def display(self, save=True):
        fs = self.data["fs"]
        gs = self.data["gs"]
        i = self.data["I"]
        q = self.data["Q"]
        mag = np.abs(i + 1j * q)
        fig = plt.figure(figsize=(9,7))
        cbar = plt.pcolor(gs, fs, mag)
        plt.colorbar()
        cbar.set_label("Amplitude (ADC Units)")
        plt.title("Gain Rabi + Pulse Frequency")
        plt.xlabel("Gain")
        plt.ylabel("Qubit Pulse Frequency (MHz)")
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return
        
        





