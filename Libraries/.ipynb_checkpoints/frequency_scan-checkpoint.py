# a tProc configured ADC channel is required here
class SinglePulseFreqLoop(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        gen_ch = cfg['gen_ch']
        
        self.declare_gen(ch=gen_ch, nqz=1)
        self.declare_readout(ch=ro_ch, length=cfg['ro_len'])

        self.add_loop(name='freq_loop', count=cfg['steps'])
        self.add_readoutconfig(ch=ro_ch, name='ro', freq=cfg['freq'], gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="myconst", ro_ch=ro_ch, 
                       style="const", 
                       length=cfg['pulse_len'], 
                       freq=cfg['freq'], 
                       phase=0,
                       gain=1.0,
                      )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name='ro', t=0)
        self.pulse(ch=cfg['gen_ch'], name="myconst", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'], ddr4=True)

start = 5000
stop = 6000
step = 2
fs = np.arange(start, stop, step)

pulse_len = 1

config = {'gen_ch': 0,
          'ro_ch': 0,
          'freq': QickSweep1D('freq_loop', start, stop),
          'trig_time': TRIG_TIME,
          'pulse_len': pulse_len,
          'ro_len': pulse_len,
          'steps': 500
         }



prog = SinglePulseFreqLoop(soccfg, reps=180, final_delay=0.5, cfg=config)

iq_list = prog.acquire(soc)
data = np.abs(iq_list[0][0].dot([1,1j]))

plt.scatter(fs, data)
plt.xlabel("Frequency, MHz")
plt.ylabel("a.u.")
plt.title("Frequency Spectroscopy")
plt.show()
