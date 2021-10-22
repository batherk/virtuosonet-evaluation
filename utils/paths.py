import os

def get_current_directory_path():
    directory_path, filename = os.path.split(__file__)
    return os.path.join(directory_path, "")
