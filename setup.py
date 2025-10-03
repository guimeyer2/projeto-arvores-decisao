from setuptools import setup, find_packages

setup(
    name='pacote_arvores',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    author='Guilherme Meyer',
    author_email='guimeygui@gmail.com',
    description='Implementação dos algoritmos ID3, C4.5 e CART.'
)