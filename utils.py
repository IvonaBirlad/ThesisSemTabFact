import os


def get_parent_directory_path(steps):
    """
    Get the absolute path to the parent directory of the current file.

    Returns:
        file_path (str): The absolute path to the parent directory of the current file.
    """
    parent = os.sep.join(os.getcwd().split(os.sep)[:-steps])
    return parent

