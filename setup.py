from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="partivisionML",
    version="0.17",
    packages=find_packages(where="src"),
    package_dir= {"": "src"},
    package_data= {"": ["*/.gitkeep", "*/default_settings.json"]},
    description=description,
    url="https://github.com/AndKaminer/PartiVisionML",
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
        'console_scripts': ["pv-app = partivision.app:main",
                            "pv-set-settings = partivision.settings:set_new_settings_file",
                            "pv-get-settings-path = partivision.settings:print_settings_file_path",
                            "pv-get-setting = partivision.settings:print_setting"]}
)
