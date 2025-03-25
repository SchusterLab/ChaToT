from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

class single_shot_excited(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.ro_len)

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
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.pulses.pi_const.length, ddr4=False)

class excited_measurement(Experiment):
    def __init__(self, path='', prefix='excited_measurement', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        return

    def acquire(self, progress=False):
        super().acquire()
        prog = single_shot_excited(soccfg=self.soccfg, reps=1, cfg=self.cfg, final_delay=None)
        iq_list = prog.acquire_decimated(self.soc, soft_avgs=self.cfg.expt.n_avg)
        t = prog.get_time_axis(ro_index=0)
        data = {"I": iq_list[0][:,0], "Q": iq_list[0][:,1], "time": t}
        self.data = data
        return data

    def display(self, save=True):
        data = self.data
        t = data["time"]
        i = data["I"]
        q = data["Q"]
        mag = np.abs(i + 1j * q)
        fig = plt.figure(figsize=(9,7))
        plt.plot(t, mag, label="magnitude")
        plt.plot(t, i, label="i")
        plt.plot(t, q, label="q")
        plt.legend()
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.title("Time of Flight (after a pi pulse)\nFrequency: " + str(self.cfg.expt.res_freq))
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return







