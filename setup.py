from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="golgi-cell-cv",
    version="0.7",
    packages=find_packages(where="src"),
    package_dir= {"": "src"},
    package_data= {"": ["*/.gitkeep"]},
    description=description,
    url="https://github.com/AndKaminer/golgi",
    author="Andrew Kaminer",
    author_email="akaminer@gatech.edu",
    install_requires=[
        'opencv-python',
        'numpy',
        'roboflow',
        'easygui',
        'huggingface_hub',
        'ultralytics',
        'diplib'
    ],
    entry_points={
        'console_scripts': ["golgi-train = golgi.training:main",
                            "golgi-track = golgi.inference:main",
                            "golgi-weights-list = golgi.inference:list_weights",
                            "golgi-weights-download = golgi.inference:download_weights",
                            "golgi-annotate = golgi.annotation:main"]}
)
