import pickle
from pathlib import Path


def save_state_dict(state_dict, filepath):
    """
    Saves a model's state_dict to a file.

    Args:
        state_dict (dict): Dictionary mapping parameter names to NumPy arrays.
        filepath (str or Path): The path to save the weights to.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"Model state saved to {filepath}")


def load_state_dict(filepath):
    """
    Loads a model's state_dict from a file.

    Returns:
        dict: The loaded state_dict.
    """
    with open(filepath, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict
