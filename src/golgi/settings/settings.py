import os
import json
import pathlib
import sys
import argparse


DEFAULT_SETTINGS_PATH = os.path.join(pathlib.Path(__file__).parents[0], "default_settings.json") # NEVER LET USER MODIFY THIS
NORMAL_SETTINGS_PATH = os.path.join(pathlib.Path(__file__).parents[0], "settings.json") # user can modify this
SETTINGS_PATH_PATH = os.path.join(pathlib.Path(__file__).parents[0], "settings_path.txt") # user can modify this, but should only do so through commands we provide


settings = None
settings_path = None


class SettingNotFoundException(Exception):
    def __init__(self, setting, message=None):
        if message == None:
            message = f"Setting '{setting}' not found in the settings.json file."

        super().__init__(message)

class SettingNotSetException(Exception):
    def __init__(self, setting, message=None):
        if message == None:
            message = f"Setting '{setting}' is not set. Please go to your settings file to set it."

        super().__init__(message)


def read_settings(path):
    global settings
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                settings = json.load(f)
        except Exception:
            raise Exception("Could not read settings file. Are you sure it's a valid JSON?")
    else:
        raise FileNotFoundError("Settings file not found")

def get_setting(setting_name):
    if setting_name not in settings:
        raise SettingNotFoundException(setting_name)

    if settings[setting_name] == None:
        raise SettingNotSetException(setting_name)

    return settings[setting_name]

def soft_get_setting(setting_name, default=None):
    out = settings.get(setting_name, default)
    return out if out is not None else default

def get_settings_path(settings_path_path):
    try:

        if not os.path.exists(settings_path_path): # if path to settings path file doesn't exist, create it with reference to the standard settings file (also create said standard file)
            with open(settings_path_path, "w") as f:
                f.write(NORMAL_SETTINGS_PATH)

            with open(DEFAULT_SETTINGS_PATH, "r") as default:
                with open(NORMAL_SETTINGS_PATH, "w") as normal:
                    temp_settings = json.load(default)
                    normal.write(json.dumps(temp_settings, indent=4))

        with open(settings_path_path, "r") as f:
            settings_path = f.read()

        if not os.path.exists(settings_path):
            with open(DEFAULT_SETTINGS_PATH, "r") as default:
                temp_settings = json.load(default)
                with open(settings_path, "w") as normal:
                    normal.write(json.dumps(temp_settings, indent=4))

        return settings_path

    except Exception:
        raise Exception("Could not find settings file")

def restore_default_settings():
    # change settings_path to the standard settings file.
    # reset the standard settings file.
    # reset the settings object

    global settings
    global settings_path

    with open(SETTINGS_PATH_PATH, "w") as f:
        f.write(NORMAL_SETTINGS_PATH)

    settings_path = NORMAL_SETTINGS_PATH

    with open(DEFAULT_SETTINGS_PATH, "r") as f:
        settings = json.load(f)

    with open(NORMAL_SETTINGS_PATH, "w") as f:
        f.write(json.dumps(settings, indent=4))

def set_new_settings_file(argv=sys.argv):
    global settings_path

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=os.path.abspath, help='Enter the path to the settings file')
    args = parser.parse_args(argv[1:])
    path = args.path

    if not os.path.exists(path):
        raise FileNotFoundError("Settings file not found")

    with open(SETTINGS_PATH_PATH, "w") as file:
        file.write(path)

    settings_path = get_settings_path(SETTINGS_PATH_PATH)

    read_settings(settings_path)

def print_settings_file_path():
    global settings_path
    print(settings_path)

def print_setting(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', type=str, help="Setting to print")
    args = parser.parse_args(argv[1:])
    setting = args.setting

    print(get_setting(setting))


settings_path = get_settings_path(SETTINGS_PATH_PATH)
read_settings(settings_path)
