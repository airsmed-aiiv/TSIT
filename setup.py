from setuptools import setup, find_packages

setup(
    name            = 'tsit',
    version         = '0.1',
    description     = 'TSIT implementation in PyTorch; TSIT: A Simple and Versatile Framework for Image-to-Image Translation',
    author          = 'Younghan Kim',
    author_email    = 'godppkyh@artiq.kr',
    install_requires= [],
    packages        = find_packages(),
    python_requires = '>=3.9.0'  
)