from slab import Experiment, AttrDict
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py
import numpy as np
import matplotlib.pyplot as plt

class res_power_spec_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        
        self.declare_gen(ch=self.cfg.soc.res_gen_ch, nqz=2)
        self.declare_readout(ch=self.cfg.soc.ro_ch, length=self.cfg.expt.ro_len)

        self.add_loop(name='gain_loop', count=self.cfg.expt.gain_steps)
        self.add_loop(name='freq_loop', count=self.cfg.expt.freq_steps)
    
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
        

class resonator_power_spectroscopy(Experiment):
    # import config file, specify data path
    def __init__(self, path='', prefix='resonator_power_spectroscopy', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        
    # run the experiment
    def acquire(self, progress=False):
        super().acquire()
        prog = res_power_spec_pulse(soccfg=self.soccfg, reps=self.cfg.expt.n_avg, final_delay=0.5, cfg=self.cfg)
        iq_list = prog.acquire(self.soc)
        fs = np.linspace(self.cfg.expt.center - self.cfg.expt.span, self.cfg.expt.center + self.cfg.expt.span, self.cfg.expt.freq_steps)
        gs = np.linspace(self.cfg.expt.min_gain, self.cfg.expt.max_gain, self.cfg.expt.gain_steps)
        data = {"I": iq_list[0][0][:,:,0], "Q": iq_list[0][0][:,:,1], "fs": fs, "gs": gs}
        self.data = data
        return data

    # plot results
    def display(self, save=True):
        data = self.data
        fs = data["fs"]
        gs = data["gs"]
        i = data["I"]
        q = data["Q"]
        mag = np.abs(i + 1j * q)
        for i in range(len(gs)):
            mag[i] -= np.mean(mag[i])
        # col_sums = mag.sum(axis=1)
        # mag /= col_sums[:, np.newaxis]
        # mag_norm = mag
        # for iq in range(len(mag_norm)):
        #     mag_norm[iq] = mag_norm[iq] / mag_norm[iq][0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 14))
        pc = ax1.pcolor(fs, gs, mag)
        cbar = plt.colorbar(pc, ax=ax1)
        cbar.set_label("a.u.")
        ax1.set_xlabel("frequency (MHz)")
        ax1.set_ylabel("gain")
        ax2.set_xlabel("frequency (MHz)")
        ax2.set_ylabel("a.u. (normalized)")
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.cfg.expt.gain_steps))
        for i, color in enumerate(colors):
            ax2.plot(fs, mag[i], "-o", color=color, label=gs[i])
        ax2.legend()
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return

