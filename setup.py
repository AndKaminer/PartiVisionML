from setuptools import find_packages, setup

setup(
        name="Cell CV Lib",
        version="1.0",
        packages=find_packages(),
        install_requires=[
            'opencv-python',
            'numpy']
        )
