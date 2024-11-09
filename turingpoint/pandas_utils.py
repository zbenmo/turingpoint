import pandas as pd


class ReplayBufferCollector:
    """This is a collector based on a pandas DataFrame. It is a participant that just copies entries.
    It is meant to be used as a replay buffer with a simple logic that keeps the newest entries.
    """
    def __init__(self, collect, max_entries=10_000):
        self.collect = list(collect)
        self.max_entries = max_entries
        self.replay_buffer = None

    def __call__(self, parcel: dict):
        self.replay_buffer = pd.concat([
            self.replay_buffer, 
            pd.DataFrame.from_records([{
                k: parcel[k] for k in self.collect
            }])
        ])
        self.replay_buffer = self.replay_buffer.tail(self.max_entries)
