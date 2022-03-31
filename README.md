[![DOI](https://zenodo.org/badge/209026254.svg)](https://zenodo.org/badge/latestdoi/209026254)
# py_multislice

![](cbed.png)

Python multislice slice code

GPU accelerated using 
[pytorch](https://pytorch.org/)

Ionization based off [Flexible Atomic Code (FAC)](https://github.com/flexible-atomic-code/fac).

# Installation

1. Clone or branch this repo into a directory on your computer

```bash
    $ git clone https://github.com/HamishGBrown/py_multislice.git
```

2. (Optional) create a new conda environment for py_multislice:

```bash
    $ conda create --name py_multislice
    $ conda activate py_multislice
```

3. In the command-line (Linux or Mac) or your Python interpreter (Windows) install pytorch, you will need to choose the conda version appropriate for your GPU (or CPU only version if required) see [here](https://pytorch.org/get-started/locally/), and in the root directory of your local copy of the repo run the install command for py_multislice

```bash
    $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    $ pip install -e .
```

   All necessary dependencies should also be installed, if you have issues try installing in a fresh anaconda environment (step 2).

4. If you would like to perform ionization based TEM simulations, download and install the flexible atomic code (FAC), including the python interface (pfac), from [here](https://github.com/flexible-atomic-code/fac). I've only successfully got this working on Linux, your mileage may vary on Windows operating systems. 

5. As an added precaution, run the Test.py script to ensure everything is working as expected

```bash
    $ python Test.py
```

    If you didn't instal PFAC in the last step then you will get error messages from the ionization routines, if you only want to perform non-ionization based TEM simulations then you can ignore these failed tests. You can also run run Orbital_normalization.py to test that the ionization cross-sections are being calculated appropriately.

# Documentation and demos

Documentation can be found [here](https://hamishgbrown.github.io/py_multislice/pyms/), for demonstrations and walk throughs on common simulation types see the Jupyter Notebooks in the [Demos](Demos/) folder. The Notebook for STEM-EELS is still under construction.

# Bug-fixes and contributions

Message me, leave a bug report and fork the repo. All contributions are welcome.

# Acknowledgements

A big thanks to [Philipp Pelz](https://github.com/PhilippPelz) for teaching me the ins and outs of pytorch and numerous other discussions on computing and electron microscopy. Credit to [Colin Ophus](https://github.com/cophus) for many discussions and much inspiration re the PRISM algorithm (of which he is the inventor). Thanks to my boss Jim Ciston for tolerating this side project! Thankyou to [Thomas Aarholt](https://github.com/thomasaarholt) for Python advice and testing of different libraries.  Thanks to Adrian D'Alfonso, Scott Findlay and Les Allen (my PhD advisor) for originally teaching me the art of multislice and ionization.


