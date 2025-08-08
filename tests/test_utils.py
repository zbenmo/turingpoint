import unittest
from turingpoint.utils import ReplayBufferCollector


class TestReplayBufferCollector(unittest.TestCase):
    def test_max_entries_limit(self):
        """Test that ReplayBufferCollector respects the max_entries limit and keeps newest entries."""
        # Create a collector with max 5 entries
        collector = ReplayBufferCollector(['obs', 'action', 'reward'], max_entries=5)
        
        # Add 7 entries (more than the max_entries limit)
        for i in range(7):
            parcel = {
                'obs': f'observation_{i}',
                'action': f'action_{i}',
                'reward': i,
                'step': i,
                'other_field': 'should_not_be_collected'
            }
            collector(parcel)
        
        # Verify that only 5 entries are kept
        self.assertEqual(len(collector.replay_buffer), 5)
        
        # Verify that the newest entries are kept (entries 2-6, since we added 7 entries)
        expected_entries = [
            {'obs': 'observation_2', 'action': 'action_2', 'reward': 2},
            {'obs': 'observation_3', 'action': 'action_3', 'reward': 3},
            {'obs': 'observation_4', 'action': 'action_4', 'reward': 4},
            {'obs': 'observation_5', 'action': 'action_5', 'reward': 5},
            {'obs': 'observation_6', 'action': 'action_6', 'reward': 6}
        ]
        
        self.assertEqual(collector.replay_buffer, expected_entries)
    
    def test_collects_only_specified_keys(self):
        """Test that ReplayBufferCollector only collects the specified keys."""
        collector = ReplayBufferCollector(['obs', 'reward'], max_entries=3)
        
        parcel = {
            'obs': 'test_observation',
            'action': 'test_action',
            'reward': 1.5,
            'step': 0,
            'extra_field': 'should_not_be_collected'
        }
        
        collector(parcel)
        
        # Verify only specified keys are collected
        expected_entry = {'obs': 'test_observation', 'reward': 1.5}
        self.assertEqual(collector.replay_buffer[0], expected_entry)
    
    def test_empty_buffer_initially(self):
        """Test that the replay buffer starts empty."""
        collector = ReplayBufferCollector(['obs'], max_entries=5)
        self.assertEqual(len(collector.replay_buffer), 0)
    
    def test_does_not_exceed_max_entries(self):
        """Test that the buffer never exceeds max_entries."""
        collector = ReplayBufferCollector(['obs'], max_entries=3)
        
        # Add exactly max_entries
        for i in range(3):
            parcel = {'obs': f'obs_{i}'}
            collector(parcel)
        
        self.assertEqual(len(collector.replay_buffer), 3)
        
        # Add one more entry
        parcel = {'obs': 'obs_3'}
        collector(parcel)
        
        # Should still have only 3 entries
        self.assertEqual(len(collector.replay_buffer), 3)
        
        # Should have the newest entries
        expected_entries = [
            {'obs': 'obs_1'},
            {'obs': 'obs_2'},
            {'obs': 'obs_3'}
        ]
        self.assertEqual(collector.replay_buffer, expected_entries)


if __name__ == '__main__':
    unittest.main() 