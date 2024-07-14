from setuptools import find_packages, setup

setup(
    name="golgi",
    version="1.0",
    packages=find_packages(where="golgi/*"),
    install_requires=[
        'opencv-python',
        'numpy',
        'roboflow',
        'easygui',
        'huggingface_hub',
        'ultralytics'
    ]
)
