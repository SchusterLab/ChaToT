"""
PLEASE READ:

This program is different from resonator_spectroscopy.py. In resonator_spectroscopy.py, frequency is swept
over once per loop, and these single frequency measurement sweeps are then averaged together. This may potentially
introduce inaccuracy in resonance measurements, since in many practical experiments the resonance is repeatedly
measured, and due to thermal effects this repeated measurement may cause what the resonance is for the duration
of the experiment to be different than the measured resonance from spectroscopy. To remedy this, this program
sits on a frequency, taking many measurements and averaging them, before moving on to the next frequency, so 
frequency is only swept through once with all the averages being taken over the course of this single iteration.
In other words, it performs the two loops, the averaging loop and the frequency sweeping loop, in reverse order.
Since QICK automatically averages over the reps axis regardless if its the outermost or innermost loop,
software averaging must be used.

"""

from slab import Experiment, AttrDict
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../Utilities")
from resonance_fitting import *

class res_spec_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        
        self.declare_gen(ch=self.cfg.soc.res_gen_ch, nqz=2)
        self.declare_readout(ch=self.cfg.soc.ro_ch, length=self.cfg.expt.pulse_len)
    
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
        self.pulse(ch=self.cfg.soc.res_gen_ch, name="pulse", t=0)
        self.trigger(ros=[self.cfg.soc.ro_ch], pins=[0], t=self.cfg.expt.trig_offset, ddr4=False)
        
        

class resonator_spectroscopy(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='resonator_spectroscopy_reverse_order', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        super().acquire()
        fs = np.linspace(self.cfg.expt.center - self.cfg.expt.span, self.cfg.expt.center + self.cfg.expt.span, self.cfg.expt.steps)
        iq_list = []
        for f in fs:
            self.cfg.expt.freq = f
            prog = res_spec_pulse(soccfg=self.soccfg, reps=self.cfg.expt.n_avg, final_delay=0.5, cfg=self.cfg)
            iq = prog.acquire(self.soc, progress=progress)
            iq_list.append(iq)
        data = {"I": np.array(iq_list)[:,0,0,0], "Q": np.array(iq_list)[:,0,0,1], "fs": fs}
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
        plt.plot(fs, mag, "-o")
        # plt.plot(i, q)
        plt.ylabel("a.u.")
        plt.xlabel("MHz")
        # plt.xlabel("I")
        # plt.ylabel("Q")
        plt.title("Resonator Spectroscopy (Reverse Order)")
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return

    def fit(self, plot=False): # this doesnt work yet
        fs = self.data['fs']
        I = self.data['I']
        Q = self.data['Q']
        S = I + 1j*Q
        R = np.abs(S)

        # guess resonance frequency
        f0_guess = fs[np.argmin(R)]

        fit_params = roughfit(fs, S, f0_guess, plot=plot)
        print(fit_params)

