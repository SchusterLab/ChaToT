"""
Author: Charles Snipp
Last Updated: 10/21/24

This class defines programs that can be run to determine the time of flight
of output pulses in order to calibrate the ADC trigger offset

cfg.expt = {
    "n_avg": 500,
    "ro_len": 13,
    "pulse_len": 7,
    "freq": 5981.7,
    "phase": 0,
    "gain": 1,
    "trig_offset": 0
}

"""

from slab import Experiment, AttrDict
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2
import os, h5py
import numpy as np
import matplotlib.pyplot as plt
from slab.instruments import InstrumentManager

class tof_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        
        self.declare_gen(ch=self.cfg.soc.res_gen_ch, nqz=2)
        self.declare_readout(ch=self.cfg.soc.ro_ch, length=self.cfg.expt.ro_len)
        self.add_readoutconfig(ch=self.cfg.soc.ro_ch, name='ro', freq=self.cfg.expt.freq, gen_ch=self.cfg.soc.res_gen_ch)

        self.add_pulse(ch=self.cfg.soc.res_gen_ch, name="pulse", ro_ch=self.cfg.soc.ro_ch, 
                       style="const", 
                       length=self.cfg.expt.pulse_len,
                       freq=self.cfg.expt.freq, 
                       phase=self.cfg.expt.phase,
                       gain=self.cfg.expt.gain, 
                      )

    def _body(self, cfg):
        self.send_readoutconfig(ch=self.cfg.soc.ro_ch, name='ro', t=0)
        self.trigger(ros=[self.cfg.soc.ro_ch], pins=[0], t=self.cfg.expt.trig_offset, ddr4=False)
        self.pulse(ch=self.cfg.soc.res_gen_ch, name="pulse", t=0)
        

class time_of_flight(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='time_of_flight', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        super().acquire()
        prog = tof_pulse(soccfg=self.soccfg, reps=1, final_delay=None, cfg=self.cfg)
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
        plt.title("Time of Flight\nFrequency: " + str(self.cfg.expt.freq))
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return
        
        
        
        



    