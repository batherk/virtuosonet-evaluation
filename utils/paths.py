import os
from pathlib import Path

def get_current_directory_path():
    directory_path, filename = os.path.split(__file__)
    return os.path.join(directory_path, "")

def get_root_folder():
    return os.path.join(Path(__file__).parent.parent, '')
