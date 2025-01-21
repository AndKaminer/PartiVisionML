from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="golgi-cell-cv",
    version="0.12",
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
        'diplib',
        'boto3',
        'dash',
        'dash-bootstrap-components'
    ],
    entry_points={
        'console_scripts': ["golgi-app = golgi.app:main"  ]}
)
