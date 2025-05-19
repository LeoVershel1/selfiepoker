from setuptools import setup, find_packages

setup(
    name="simple_game",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask==3.0.2',
        'flask-cors==4.0.0',
    ],
) 