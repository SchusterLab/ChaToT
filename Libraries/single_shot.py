from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

# single shot:
# prepare in ground state, measure, do n times
# prepare in excited state by measuring pi pulse, measure, do n times

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
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.pulses.pi_const.length+0.01, ddr4=True)


class single_shot(Experiment):
    def __init__(self, path='', prefix='single_shot', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        return

    def acquire(self, progress=False):
        super().acquire()
        
        prog_ground = single_shot_ground(soccfg=self.soccfg, reps=1, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        print(self.soc)
        iq_list_ground = prog_ground.acquire(self.soc)
        data_g = {"I": iq_list_ground[0][0][:,0], "Q": iq_list_ground[0][0][:,1]}
        
        prog_excited = single_shot_excited(soccfg=self.soccfg, reps=1, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        iq_list_excited = prog_excited.acquire(self.soc)
        data_e = {"I": iq_list_excited[0][0][:,0], "Q": iq_list_excited[0][0][:,1]}

        self.data = {"ground": data_g, "excited": data_e}
        return self.data

    def display(self, save=True): # TODO: rotating data, finding mean, plotting projection
        i_g = self.data["ground"]["I"]
        q_g = self.data["ground"]["Q"]
        i_e = self.data["excited"]["I"]
        q_e = self.data["excited"]["Q"]
        fig = plt.figure(figsize=(9,7))
        plt.plot(i_g, q_g, '.', color='blue', label='g')
        plt.plot(i_e, q_e, '.', color='red', label='e')
        plt.legend()
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return