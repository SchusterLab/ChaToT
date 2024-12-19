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

        ro_ch = cfg.soc.ro_ch
        qubit_gen_ch = cfg.soc.qubit_gen_ch
        res_gen_ch = cfg.soc.res_gen_ch
        res_freq = cfg.expt.res_freq
        res_gain = cfg.expt.res_gain
        res_pulse_len = cfg.expt.res_pulse_len
        res_phase = cfg.expt.res_phase
        qubit_gain = cfg.expt.qubit_gain
        qubit_freq = cfg.expt.qubit_freq
        qubit_pulse_len = cfg.expt.qubit_pulse_len
        qubit_phase = cfg.expt.qubit_phase
        trig_offset = cfg.expt.trig_offset
        relaxation_time = cfg.expt.relaxation_time
        steps = cfg.expt.steps

        self.declare_gen(ch=res_gen_ch, nqz=1)
        self.declare_gen(ch=qubit_gen_ch, nqz=1)
        self.declare_readout(ch=ro_ch, length=res_pulse_len)

        self.add_loop(name="freq_loop", count=steps)

        self.add_readoutconfig(ch=ro_ch, name='ro', freq=res_freq, gen_ch=res_gen_ch)

        self.add_pulse(ch=res_gen_ch, name="res_pulse", ro_ch=ro_ch, 
                       style="const", 
                       length=res_pulse_len,
                       freq=res_freq, 
                       phase=res_phase,
                       gain=res_gain, 
                      )

        # gaussian qubit pulses exceed buffer length for some reason

        self.add_gauss(ch=qubit_gen_ch, name="ramp", sigma=qubit_pulse_len/10, length=qubit_pulse_len, even_length=True)

        self.add_pulse(ch=qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
                       style="arb", 
                       envelope="ramp", 
                       freq=qubit_freq, 
                       phase=qubit_phase,
                       gain=qubit_gain, 
                      )

        # self.add_pulse(ch=qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
        #                style="const", 
        #                length=qubit_pulse_len,
        #                freq=qubit_freq, 
        #                phase=qubit_phase,
        #                gain=qubit_gain, 
        #               )

    def _body(self, cfg):
        cfg = AttrDict(cfg)
        
        ro_ch = cfg.soc.ro_ch
        res_gen_ch = cfg.soc.res_gen_ch
        qubit_gen_ch = cfg.soc.qubit_gen_ch
        trig_offset = cfg.expt.trig_offset
        qubit_pulse_len = cfg.expt.qubit_pulse_len
        
        self.send_readoutconfig(ch=ro_ch, name='ro', t=0)
        self.pulse(ch=qubit_gen_ch, name="qubit_pulse", t=0)
        self.pulse(ch=res_gen_ch, name="res_pulse", t=qubit_pulse_len+1)
        self.trigger(ros=[ro_ch], pins=[0], t=trig_offset+qubit_pulse_len, ddr4=False)


class qubit_twotone_spectroscopy(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='qubit_twotone_spectroscopy', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        cfg = self.cfg

        ns_address = cfg.instrument_manager.ns_address
        ns_port = cfg.instrument_manager.ns_port
        proxy_name = cfg.instrument_manager.proxy_name
        n_avg = cfg.expt.n_avg
        center = cfg.expt.center
        span = cfg.expt.span
        steps = cfg.expt.steps
        relaxation_time = cfg.expt.relaxation_time
        
        fs = np.linspace(center - span, center + span, steps)

        soc, soccfg = make_proxy(ns_host=ns_address, ns_port=ns_port, proxy_name=proxy_name)
        
        prog = twotone_pulse(soccfg=soccfg, reps=n_avg, final_delay=relaxation_time, cfg=cfg)
        iq_list = prog.acquire(soc)
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

    def savedata(self): # I should really find a better way to do all this that is more friendly with different ways of saving/storing data and configs in SLab
                        # I should also turn this into a callable function rather than having it pasted in all my files
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
        if hasattr(self,'cfg'):
            if 'expt' in self.cfg:
                for item in self.cfg.expt:
                    if isinstance(self.cfg.expt[item], QickParam):
                        self.cfg.expt[item] = 'QickSweep values are stored in data'
        with open(self.path + "cfg.json", "w") as cfg_file:
            json.dump(self.cfg, cfg_file, indent=4)
        print("Config saved to " + self.path + "cfg.json")
