"""Setup script for pyms, ensure that all dependent packages are installed."""
from setuptools import setup, find_packages

setup(
    name="pyms",
    version="0.1",
    description="An open-source Python multislice package",
    author="Hamish Brown",
    author_email="hamishgallowaybrown@gmail.com",
    url="https://github.com/HamishGBrown/py_multislice/",
    packages=find_packages(),
    install_requires=[
        "h5py >= 2.10",
        "ipython >= 4.0",
        "scipy >= 1.1",
        "matplotlib >= 3.0",
        "nbstripout",
        "numpy >= 1.17",
        "Pillow >= 6.0",
        "torch >= 1.8",
        "tqdm >= 4.48",
        "ase",
        "pypng",
        "ipywidgets",
    ],
)
