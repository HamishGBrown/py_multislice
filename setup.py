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
        'ipython >= 4.0',
        'scipy >= 1.1',
        'matplotlib >= 3.0',
        'numpy >= 1.15',
        'Pillow >= 6.0',
        'torch >= 1.2',
        'tqdm >= 4.0'
        ]

)
