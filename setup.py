from setuptools import setup, find_packages

setup(
    name='yartsev',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'pandas',
        'hdf5storage',
        'numpy==1.26.4',
        'tqdm',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'requests',
        'h5py',
        'complexnn',
        'keras-complex'
    ],
    include_package_data=True,
)