from setuptools import setup, find_packages

setup(
    name='humanoid_rl',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'mediapipe',
        'gymnasium',
        'pybullet',
        'numpy',
        'torch',
        'opencv-contrib-python',
        'Pillow',
        'matplotlib',
        'scipy',
    ],
)