from slab import experiment, AttrDict
from slab.experiment import Experiment # ?
from qick.pyro import make_proxy
from qick import *
from qick.asm_v2 import AveragerProgramV2, QickParam
import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class rabi_pulse(AveragerProgramV2):
    def _initialize(self, cfg):
        cfg = AttrDict(cfg)

        self.declare_gen(ch=cfg.soc.res_gen_ch, nqz=2)
        self.declare_gen(ch=cfg.soc.qubit_gen_ch, nqz=2)
        self.declare_readout(ch=cfg.soc.ro_ch, length=cfg.expt.res_pulse_len)

        self.add_loop(name="freq_loop", count=cfg.expt.freq_steps)
        self.add_loop(name="length_loop", count=cfg.expt.length_steps)

        self.add_readoutconfig(ch=cfg.soc.ro_ch, name='ro', freq=cfg.expt.res_freq, gen_ch=cfg.soc.res_gen_ch)

        self.add_pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", ro_ch=cfg.soc.ro_ch, 
                       style="const", 
                       length=cfg.expt.res_pulse_len,
                       freq=cfg.expt.res_freq, 
                       phase=cfg.expt.res_phase,
                       gain=cfg.expt.res_gain, 
                      )

        # constant qubit pulse
        self.add_pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
                       style="const", 
                       length=cfg.expt.qubit_pulse_len,
                       freq=cfg.expt.qubit_freq, 
                       phase=cfg.expt.qubit_phase,
                       gain=cfg.expt.qubit_gain, 
                      )
        
        # gaussian qubit pulse
        # self.add_gauss(ch=cfg.soc.qubit_gen_ch, name="ramp", sigma=cfg.expt.qubit_pulse_len/10, length=cfg.expt.qubit_pulse_len, even_length=True)
        # self.add_pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", ro_ch=None, 
        #                style="arb", 
        #                envelope="ramp", 
        #                freq=cfg.expt.qubit_freq, 
        #                phase=cfg.expt.qubit_phase,
        #                gain=cfg.expt.qubit_gain, 
        #               )

    def _body(self, cfg):
        cfg = AttrDict(cfg)
        
        self.send_readoutconfig(ch=cfg.soc.ro_ch, name='ro', t=0)
        self.pulse(ch=cfg.soc.qubit_gen_ch, name="qubit_pulse", t=0)
        self.pulse(ch=cfg.soc.res_gen_ch, name="res_pulse", t=cfg.expt.qubit_pulse_len+0.01)
        self.trigger(ros=[cfg.soc.ro_ch], pins=[0], t=cfg.expt.trig_offset+cfg.expt.qubit_pulse_len, ddr4=False)


class rabi_length_sweep(Experiment):
    def __init__(self, path='', prefix='rabi_length_sweep', config_file=None, liveplot_enabled=True, **kwargs):
        super().__init__(path=path, prefix=prefix, config_file=config_file, liveplot_enabled=liveplot_enabled, **kwargs)
        return

    def acquire(self, progress=False):
        super().acquire()
        prog = rabi_pulse(soccfg=self.soccfg, reps=self.cfg.expt.n_avg, final_delay=self.cfg.expt.relaxation_time, cfg=self.cfg)
        fs = np.linspace(self.cfg.expt.min_freq, self.cfg.expt.max_freq, self.cfg.expt.freq_steps)
        ls = np.linspace(self.cfg.expt.min_length, self.cfg.expt.max_length, self.cfg.expt.length_steps) # relaxation time should be somewhere else in the cfg
        iq_list = prog.acquire(self.soc)
        self.data = {"I": iq_list[0][0][:,:,0], "Q": iq_list[0][0][:,:,1],"fs": fs, "ls": ls}
        return self.data

    def display(self, save=True):
        fs = self.data["fs"]
        ls = self.data["ls"]
        i = self.data["I"]
        q = self.data["Q"]
        mag = np.abs(i + 1j * q)
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.cfg.expt.freq_steps))
        fig = plt.figure(figsize=(9,7))
        plt.pcolor(ls, fs, mag)
        cbar = plt.colorbar()
        cbar.set_label("Amplitude (ADC Units)")
        plt.title("Length Rabi vs Pulse Frequency")
        plt.xlabel("Length (us)")
        plt.ylabel("Qubit Pulse Frequency (MHz)")
        plt.legend()
        plt.show()
        if save:
            if not os.path.exists(self.path):
                print(f'Creating directory {self.path}')
                os.makedirs(self.path)
            fig.savefig(self.path)
        return

    def fit(self, save_plot=True, write_to_cfg=True):
        fs = self.data["fs"]
        ls = self.data["ls"]
        i = self.data["I"]
        q = self.data["Q"]
        mag = np.abs(i + 1j * q)
        def cos_wave(x, A, B, C, D):
            return A * np.cos(B*x + C) + D
        rabi_freqs = []
        for i in range(len(fs)):
            xlim = 12
            print(i)
            print(mag[i][:xlim])
            rabi_osc_params, rabi_osc_cov = curve_fit(cos_wave, ls[:xlim], mag[i][:xlim])
            rabi_freqs.append(rabi_osc_params[1])
        def parabola(x, A, B, C):
            return A*(x**2) + B*x + C
        rabi_fit_params, rabi_fit_cov = curve_fit(parabola, fs, rabi_freqs)
        xs = np.linspace(fs[0], fs[-1], 100)
        ys = parabola(xs, *rabi_fit_params)
        minimum = xs[np.argmin(ys)]
        minimum_label = 'minimum frequency: ' + str(minimum)
        fig = plt.figure(figsize=(9,7))
        print(ls)
        print(mag[24])
        # plt.plot(ls, mag[24])
        plt.plot(fs, rabi_freqs, 'o', label='rabi frequencies')
        plt.plot(xs, ys, label='fit')
        plt.axvline(x=minimum, label=minimum_label)
        plt.show()
        





