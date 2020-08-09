# py_multislice

![](cbed.png)

Python multislice slice code

GPU accelerated using 
[pytorch](https://pytorch.org/)

Ionization based off [Flexible Atomic Code (FAC)](https://github.com/flexible-atomic-code/fac), ionization potentials now working though correct units are still forthcoming!

# Installation

1. Clone or branch this repo into a directory on your computer

2. In the command-line (Linux or Mac) or your Python interpreter (Windows) run

    $ pip install -e .

   All necessary dependencies (pytorch etc.) should also be installed, if you have issues try installing in a fresh anaconda environment.

3. As an added precaution, run the Test.py script to ensure everything is working as expected

    $ python Test.py

# Documentation and demos

Documentation can be found [here](html/pyms/index.html), for demonstrations and walk throughs on common simualtion types see the Jupyter Notebooks in the [Demos](Demos/) folder.

# Bug-fixes and contributions

Message me, leave a bug report and fork the repo all contributions are welcome.

# Acknowledgements

A big thanks to [Philipp Pelz](@PhilippPelz) for teaching me the ins and outs of pytorch and numerous other discussions on computing and electron microscopy. Credit to [Colin Ophus](@cophus) for many discussions and much inspiration re the PRISM algorithm (of which he is the inventor). Thanks to my boss Jim Ciston for tolerating this side project! Thanks to Adrian D'Alfonso, Scott Findlay and Les Allen for originally teaching me the art of multislice and ionization.


