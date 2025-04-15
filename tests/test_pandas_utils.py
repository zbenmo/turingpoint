import pytest

from turingpoint.pandas_utils import (
    PDReplayBufferCollector,
)


def test_replay_buffer():
    rb = PDReplayBufferCollector(max_entries=3)

    for i in range(2):
        rb({'obs': i, 'reward': i * 10})

    assert len(rb.replay_buffer) == 2

    for i in range(2):
        rb({'obs': 200 + i, 'reward': 200 + i * 10})

    assert len(rb.replay_buffer) == 3

    assert rb.replay_buffer['obs'].to_list() == [1, 200, 201]
    assert rb.replay_buffer['reward'].to_list() == [10, 200, 210]
