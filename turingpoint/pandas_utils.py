import pandas as pd


class PDReplayBufferCollector:
    """This is a collector based on a pandas DataFrame. It is a participant that just copies entries.
    It is meant to be used as a replay buffer with a simple logic that keeps the newest entries.
    """
    def __init__(self, keys_to_collect=['obs', 'action', 'new_obs', 'reward', 'terminated'], max_entries=10_000):
        self._keys_to_collect = list(keys_to_collect)
        self._max_entries = max_entries
        self._extra_entries = [] # TODO: potentially replace with dqueue
        self._replay_buffer = None

    def __call__(self, parcel: dict):
        new_entry = {k: v for k in self._keys_to_collect if (v := parcel.get(k, None)) != None }
        self._extra_entries.append(new_entry)

    @property
    def replay_buffer(self) -> 'pd.DataFrame':
        self._replay_buffer = pd.concat([
            self._replay_buffer, # no issue if it is None
            pd.DataFrame.from_records(self._extra_entries)
        ])
        if len(self._replay_buffer) > self._max_entries:
            self._replay_buffer = self._replay_buffer.tail(self._max_entries)
        self._extra_entries.clear()
        return self._replay_buffer
