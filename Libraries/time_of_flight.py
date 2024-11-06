"""
Author: Charles Snipp
Last Updated: 10/21/24

This class defines programs that can be run to determine the time of flight
of output pulses in order to calibrate the ADC trigger offset


TODO: add data saving that actually works



"""
from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2
import os
import numpy as np
import matplotlib.pyplot as plt

# for tProc-configred outputs
class tof_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)
        
        ro_ch = cfg.soc.ro_ch
        gen_ch = cfg.soc.res_gen_ch
        ro_len = cfg.readout.ro_len
        pulse_len = cfg.readout.pulse_len
        freq = cfg.readout.freq
        phase = cfg.readout.phase
        gain = cfg.readout.gain
        trig_offset = cfg.readout.trig_offset
        
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
        trig_offset = cfg.readout.trig_offset
        
        self.send_readoutconfig(ch=ro_ch, name='ro', t=0)
        self.trigger(ros=[ro_ch], pins=[0], t=trig_offset, ddr4=True)
        self.pulse(ch=gen_ch, name="pulse", t=0)
        

class time_of_flight(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='time_of_flight', config_file=None, liveplot_enabled=True, **kwargs):
        if not os.path.exists(path):
            print(f'Creating directory {path}')
            os.makedirs(path)
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        print(self.fname)
        print(self.prefix)
        print(self.path)
        
    # run the experiment
    def acquire(self, progress=False, save=True):
        cfg = self.cfg

        ns_address = cfg.instrument_manager.ns_address
        ns_port = cfg.instrument_manager.ns_port
        proxy_name = cfg.instrument_manager.proxy_name

        soc, soccfg = make_proxy(ns_host=ns_address, ns_port=ns_port, proxy_name=proxy_name)
        
        prog = tof_pulse(soccfg=soccfg, reps=1, final_delay=None, cfg=cfg)
        iq_list = prog.acquire_decimated(soc, soft_avgs=cfg.readout.n_avg)
        t = prog.get_time_axis(ro_index=0)
        data = {"I": iq_list[0][:,0], "Q": iq_list[0][:,1], "time": t}
        self.data = data
        # TODO: add data saving w/ h5 (after figuring out what Experiment saves by default)
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
        plt.title("Time of Flight\nFrequency: " + str(self.cfg.readout.freq))
        plt.show()
        if save: fig.savefig(self.path)
        return

    def savedata(self):
        with self.datafile() as f:
            for k, d in data.items():
                f.add(k, np.array(d))
        
        



    