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
from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt

# for tProc-configred outputs
class tof_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)
        
        ro_ch = cfg.soc.ro_ch
        gen_ch = cfg.soc.res_gen_ch
        ro_len = cfg.expt.ro_len
        pulse_len = cfg.expt.pulse_len
        freq = cfg.expt.freq
        phase = cfg.expt.phase
        gain = cfg.expt.gain
        trig_offset = cfg.expt.trig_offset
        
        self.declare_gen(ch=gen_ch, nqz=1)
        self.declare_readout(ch=ro_ch, length=ro_len)
        self.add_readoutconfig(ch=ro_ch, name='ro', freq=freq, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="pulse", ro_ch=ro_ch, 
                       style="const", 
                       length=pulse_len,
                       freq=freq, 
                       phase=phase,
                       gain=gain, 
                      )

    def _body(self, cfg):

        cfg = AttrDict(cfg)
        ro_ch = cfg.soc.ro_ch
        gen_ch = cfg.soc.res_gen_ch
        trig_offset = cfg.expt.trig_offset
        
        self.send_readoutconfig(ch=ro_ch, name='ro', t=0)
        self.trigger(ros=[ro_ch], pins=[0], t=trig_offset, ddr4=True)
        self.pulse(ch=gen_ch, name="pulse", t=0)
        

class time_of_flight(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='time_of_flight', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        cfg = self.cfg

        ns_address = cfg.instrument_manager.ns_address
        ns_port = cfg.instrument_manager.ns_port
        proxy_name = cfg.instrument_manager.proxy_name

        soc, soccfg = make_proxy(ns_host=ns_address, ns_port=ns_port, proxy_name=proxy_name)
        
        prog = tof_pulse(soccfg=soccfg, reps=1, final_delay=None, cfg=cfg)
        iq_list = prog.acquire_decimated(soc, soft_avgs=cfg.expt.n_avg)
        t = prog.get_time_axis(ro_index=0)
        data = {"I": iq_list[0][:,0], "Q": iq_list[0][:,1], "time": t}
        self.data = data
        return data

    # plot results
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

    def savedata(self): # I should really find a better way to do all this that is more friendly with different ways of saving/storing data and configs in SLab
        if not os.path.exists(self.path):
            print(f'Creating directory {self.path}')
            os.makedirs(self.path)

        # save data in h5
        with h5py.File(self.path + "data.h5", "w") as h5file:
            for key, value in self.data.items():
                h5file.create_dataset(key, data=value)
        print("Data saved to " + self.path + "data.h5")

        # save config
        # I should find a better way to do this
        with open(self.path + "cfg.json", "w") as cfg_file:
            json.dump(self.cfg, cfg_file, indent=4)
        print("Config saved to " + self.path + "cfg.json")
        
        
        
        



    