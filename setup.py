from distutils.core import setup

setup(
    name='SpikeShip',
    version='1.0',
    description='Fast, unsupervised discovery of high-dimensional neural spiking patterns based on optimal transport theory',
    author='Boris Sotomayor-Gomez',
    author_email='bsotomayor92@gmail.com',
    py_modules=['spikeship.spikeship', 'spikeship.tools'], #, 'spikeship.data_formatter'],
)
