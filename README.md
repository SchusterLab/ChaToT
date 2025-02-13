# ChaToT: Charles Tprocv2 on Transmons

## How to set up a virtual environment to run things from this repository
1. create a python virtual environment by opening cmd and typing "python -m venv C:\path\to\your\env" with the actual path to your desired virtual environment location instead of C:\path\to\your\env

2. navigate to the Scripts folder in your virtual environment path by typing "cd C:\path\to\your\env\Scripts"

3. enter the virtual environment by typing "activate"

4. Install all the following packages by entering this command while inside the virtual environment:
```
pip install git+https://github.com/SchusterLab/slab, pyro4, qick, jupyter, matplotlib, h5py, scipy, tabulate
```

5. Copy the notebooks and python files from this repository into your virtual environment

6. Replace the experiment.py and instrumentmanager.py libraries in the slab library with the versions found in the Utilities folder

7. Navigate out of the Scripts folder into your virtual environment folder and start a jupyter server, navigate to the Utilities folder, and start the nameserver by running the single code cell in the "chatot_nameserver" notebook

8. Connect to the nameserver on the RFSoC board that you are using

9. You are now ready to run everything in this repository! Please message me on slack if you have any questions or are having any trouble with anything in this repository

<img src="https://github.com/SchusterLab/ChaToT/blob/2fb4c3f055dcea4de642e47a49dfd79031bbb430/9b6c156fe8f85a6b8aa0dab9e29e07cc.png" alt="chatot" width="300" />
