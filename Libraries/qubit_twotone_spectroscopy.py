from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

class twotone_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len)

        self.add_loop(name="freq_loop", count=cfg.expt.steps)

        self.add_readoutconfig(ch=cfg.soc.ro_ch, name='ro', freq=cfg.expt.res_freq, gen_ch=cfg.soc.res_gen_ch)

        self.add_pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", ro_ch=cfg.soc.ro_ch, 
                       style="const", 
                       length=cfg.expt.res_pulse_len,
                       freq=cfg.expt.res_freq, 
                       phase=cfg.expt.res_phase,
                       gain=cfg.expt.res_gain, 
                      )

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
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.expt.qubit_pulse_len+1)
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.expt.qubit_pulse_len, ddr4=False)


class qubit_twotone_spectroscopy(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='qubit_twotone_spectroscopy', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        super().acquire()
        fs = np.linspace(self.cfg.expt.center - self.cfg.expt.span, self.cfg.expt.center + self.cfg.expt.span, self.cfg.expt.steps)
        prog = twotone_pulse(soccfg=self.soccfg, reps=self.cfg.expt.n_avg, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        iq_list = prog.acquire(self.soc)
        data = {"I": iq_list[0][0][:,0], "Q": iq_list[0][0][:,1], "fs": fs}
        self.data = data
        return data

    # plot results
    def display(self, save=True):
        data = self.data
        fs = data["fs"]
        i = data["I"]
        q = data["Q"]
        mag = np.abs(i + 1j * q)
        fig = plt.figure(figsize=(9,7))
        plt.plot(fs, mag, '-o')
        plt.ylabel("a.u.")
        plt.xlabel("MHz")
        plt.title("Qubit Twotone Spectroscopy")
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return
