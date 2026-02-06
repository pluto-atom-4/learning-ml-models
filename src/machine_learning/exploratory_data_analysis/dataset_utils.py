from pathlib import Path


def get_absolute_path(relative_path):
    """ Get absolute path for file. """
    root_dir = Path(__file__).parent.parent.parent.parent
    return root_dir / "generated" / "data" / "raw" / relative_path