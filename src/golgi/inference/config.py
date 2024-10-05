import os
import pathlib
import logging

import huggingface_hub

class Configs:
    
    temp_path = os.path.join(pathlib.Path(__file__).parents[1], "temp")
    weights_path = os.path.join(pathlib.Path(__file__).parents[1], "weights")

    if not os.path.isdir(temp_path):
        os.makedir(temp_path)

    if not os.path.isdir(weights_path):
        os.makedir(weights_path)

    huggingface_repo_id = "gt-sulchek-lab/cell-tracking"

    def huggingface_login(token, write_permission=False):
        try:
            huggingface_hub.login(token=token, write_permission=write_permission)

        except Exception as e:
            logging.error(e)
            return False

        return True

    def huggingface_logout():
        try:
            huggingface_hub.logout()

        except Exception as e:
            logging.error(e)
            return False

        return True
