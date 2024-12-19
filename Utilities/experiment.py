__author__ = 'Nitrogen'

# from liveplot import LivePlotClient
# from dataserver import dataserver_client
import os.path, os # ?
import h5py, json
from qick.asm_v2 import QickParam
from qick import *

from slab import SlabFile, InstrumentManager, get_next_filename, AttrDict, LocalInstruments


class Experiment:
    """Base class for all experiments"""

    def __init__(self, path='', prefix='data', config_file=None, liveplot_enabled=True, **kwargs):
        """ Initializes experiment class
            @param path - directory where data will be stored
            @param prefix - prefix to use when creating data files
            @param config_file - parameters for config file specified are loaded into the class dict
                                 (name relative to expt_directory if no leading /)
                                 Default = None looks for path/prefix.json

            @param **kwargs - by default kwargs are updated to class dict

            also loads InstrumentManager, LivePlotter, and other helpers
        """
        self.__dict__.update(kwargs)
        self.path = path
        self.prefix = prefix
        self.cfg = None
        if config_file is not None:
            self.config_file = os.path.join(path, config_file)
        else:
            self.config_file = None
        self.im = InstrumentManager()
        # if liveplot_enabled:
        #     self.plotter = LivePlotClient()
        # self.dataserver= dataserver_client()
        self.fname = os.path.join(path, get_next_filename(path, prefix, suffix='.h5'))

        self.load_config()

    def load_config(self):
        if self.config_file is None:
            self.config_file = os.path.join(self.path, self.prefix + ".json")
        try:
            if self.config_file[:-3] == '.h5':
                with SlabFile(self.config_file) as f:
                    cfg_str = f['config']
            else:
                with open(self.config_file, 'r') as fid:
                    cfg_str = fid.read()
            self.cfg = AttrDict(json.loads(cfg_str))
        except:
            pass
        if self.cfg is not None:
            for alias, inst in self.cfg['aliases'].items():
                setattr(self, alias, self.im[inst])

    def save_config(self):
        if self.config_file[:-3] != '.h5':
            with open(self.config_file, 'w') as fid:
                json.dump(self.cfg, fid)
            self.datafile().attrs['config'] = json.dumps(self.cfg)

    def datafile(self, group=None, remote=False, data_file = None, swmr=False):
        """returns a SlabFile instance
           proxy functionality not implemented yet"""
        if data_file ==None:
            data_file = self.fname
        if swmr==True:
            f = SlabFile(data_file, 'w', libver='latest')
        elif swmr==False:
            f = SlabFile(data_file)
        else:
            raise Exception('ERROR: swmr must be type boolean')

        if group is not None:
            f = f.require_group(group)
        if 'config' not in f.attrs:
            try:
                f.attrs['config'] = json.dumps(self.cfg)
            except TypeError as err:
                print(('Error in saving cfg into datafile (experiment.py):', err))

        return f

    def go(self):
        pass

    def savedata(self):
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

    def acquire(self):
        # get soc and soccfg
        cfg = self.cfg
        self.ns_address = cfg.instrument_manager.ns_address
        self.ns_port = cfg.instrument_manager.ns_port
        self.proxy_name = cfg.instrument_manager.proxy_name
        self.im = InstrumentManager(ns_address=self.ns_address, ns_port=self.ns_port)
        self.soc = self.im[self.proxy_name]
        self.soccfg = QickConfig(self.soc.get_cfg())


