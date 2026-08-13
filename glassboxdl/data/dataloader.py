import numpy as np

from glassboxdl.core import Tensor

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class DataLoader:
    """
    Handles batching and shuffling of datasets for the training loop.

    Supports two modes for x_data (and, independently, for y_data):

    1. In-memory arrays: pass a numpy array (or list) of already-loaded
       samples. Fine for small/medium datasets that comfortably fit in RAM.

    2. Lazy file-path loading: pass a list of image file paths (strings).
       Images are decoded from disk one batch at a time instead of all at
       once, which is what you want for large image datasets (e.g. the
       ~9.7k-image Artificial Lunar Landscape dataset) that would otherwise
       need several GB of RAM just to sit as one big array.
    """

    def __init__(
        self,
        x_data,
        y_data,
        batch_size: int = 32,
        shuffle: bool = True,
        target_size=None,
        color_mode: str = "RGB",
    ):
        """
        Args:
            x_data: array-like of samples, OR a list of image file paths.
            y_data: array-like of labels, OR a list of image file paths
                (e.g. mask images) -- same two modes as x_data.
            batch_size: samples per batch.
            shuffle: reshuffle indices at the start of every epoch.
            target_size: (width, height) to resize images to when loading
                from file paths. Required if you want a fixed batch shape
                and your source images vary in size, since a batch of
                differently-sized images can't be stacked into one array.
                Ignored for non-path (already-array) inputs.
            color_mode: "RGB", "L" (grayscale), etc. -- passed to
                PIL.Image.convert() when loading from file paths. Only
                relevant for images/masks loaded via paths.
        """
        # Validate that X and Y have the same number of samples
        if len(x_data) != len(y_data):
            raise ValueError(
                f"Length mismatch: x_data has {len(x_data)} samples, but y_data has {len(y_data)} samples."
            )

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self._x_is_paths = self._looks_like_paths(x_data)
        self._y_is_paths = self._looks_like_paths(y_data)

        if (self._x_is_paths or self._y_is_paths) and not _PIL_AVAILABLE:
            raise ImportError(
                "Loading images from file paths requires Pillow. Install it with "
                "`pip install Pillow`."
            )

        self.target_size = target_size
        self.color_mode = color_mode

        # Keep paths as plain lists (no benefit to np.asarray on strings);
        # keep already-loaded data as numpy arrays for fast fancy indexing.
        self.x = list(x_data) if self._x_is_paths else np.asarray(x_data)
        self.y = list(y_data) if self._y_is_paths else np.asarray(y_data)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(x_data)

        # An array of indices [0, 1, 2, ..., n_samples - 1]
        self.indices = np.arange(self.n_samples)

    @staticmethod
    def _looks_like_paths(data):
        """True if data is a sequence of strings (file paths) rather than
        already-loaded numeric samples."""
        if len(data) == 0:
            return False
        return isinstance(data[0], str)

    def _load_batch(self, paths):
        """Decode a batch of image file paths into a single numpy array."""
        images = []
        for path in paths:
            img = Image.open(path).convert(self.color_mode)
            if self.target_size is not None:
                img = img.resize(self.target_size)
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
        # np.stack requires every image in the batch to share the same
        # shape -- either because target_size was set, or because the
        # source images are naturally uniform in size.
        return np.stack(images)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        """
        Called when a loop over the DataLoader begins.
        Shuffles the indices once per epoch if shuffle = True.
        """

        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_index = 0
        return self

    def __next__(self):
        """
        Yields the next mini-batch of Tensors.
        """

        if self.current_index >= self.n_samples:
            raise StopIteration

        # Slice the next batch of indices
        batch_idx = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]

        # Increment the index for the next call
        self.current_index += self.batch_size

        if self._x_is_paths:
            x_batch = self._load_batch([self.x[i] for i in batch_idx])
        else:
            x_batch = self.x[batch_idx]

        if self._y_is_paths:
            y_batch = self._load_batch([self.y[i] for i in batch_idx])
        else:
            y_batch = self.y[batch_idx]

        batch_x = Tensor(x_batch, requires_grad=True)
        batch_y = Tensor(y_batch, requires_grad=False)

        return batch_x, batch_y
