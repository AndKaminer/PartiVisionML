from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="golgi-cell-cv",
    version="0.15",
    packages=find_packages(where="src"),
    package_dir= {"": "src"},
    package_data= {"": ["*/.gitkeep", "*/default_settings.json"]},
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
        'console_scripts': ["golgi-app = golgi.app:main",
                            "golgi-set-settings = golgi.settings:set_new_settings_file",
                            "golgi-get-settings-path = golgi.settings:print_settings_file_path",
                            "golgi-get-setting = golgi.settings:print_setting"]}
)
