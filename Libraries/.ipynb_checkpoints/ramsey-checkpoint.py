from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

def ramsey_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)
        
        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len)

        self.add_loop(name="delay_loop", count=cfg.expt.steps)

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
        self.pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", t=cfg.expt.delay+cfg.pulses.pi_gaus.length)
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.expt.delay+(cfg.pulses.pi_gaus.length*2))
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.expt.delay+(cfg.pulses.pi_gaus.length*2), ddr4=False)





