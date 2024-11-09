from torch.utils.tensorboard import SummaryWriter


class Logging:
    """This class represent a TensorBoard file as a participant.
    Values from 'track' shall be written as scalars to TensorBoard.
    Be aware that the values are poped from the parcel.
    There is a requirement that a key 'step' is present in the parcel.
    One should use this class as a context manager, so that the file is closed when the loop is done.
    """
    def __init__(self, path: str, track):
        self.path = path
        self.track = list(track)

    def __enter__(self):
        self.writer = SummaryWriter(self.path)
        return self

    def __call__(self, parcel: dict):
        step = parcel['step']
        for key in self.track:
            val = parcel.pop(key, None)
            if val is not None:
                self.writer.add_scalar(key, val, step)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()
