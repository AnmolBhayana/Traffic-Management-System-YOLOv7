"""
pipeline.py
============
Backend pipeline for real-time traffic data processing.
Aggregates vehicle counts, calculates density, and maintains history.
"""

import time
from collections import defaultdict, deque
from datetime import datetime


# Density thresholds (vehicles per frame)
DENSITY_THRESHOLDS = {
    'LOW':    (0, 5),
    'MEDIUM': (5, 12),
    'HIGH':   (12, float('inf'))
}


class TrafficPipeline:
    """
    Processes raw detections into meaningful traffic metrics.

    Maintains a rolling window of frame data to compute:
    - Per-vehicle-type counts
    - Overall traffic density
    - Density status (Low / Medium / High)
    - Historical time-series for the dashboard
    """

    def __init__(self, history_window=60):
        """
        Parameters
        ----------
        history_window : int
            Number of recent frames to keep in rolling history
        """
        self.history_window = history_window
        self.frame_history = deque(maxlen=history_window)
        self.total_counts = defaultdict(int)
        self.total_frames = 0
        self.session_start = datetime.now()

    def process(self, detections):
        """
        Process detections from one frame.

        Parameters
        ----------
        detections : list of dict
            Output from VehicleDetector.detect()

        Returns
        -------
        metrics : dict
            Current frame metrics
        """
        counts = defaultdict(int)
        for det in detections:
            counts[det['label']] += 1

        total = sum(counts.values())
        density_status = self._get_density(total)

        frame_data = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'counts': dict(counts),
            'total': total,
            'density': density_status
        }

        self.frame_history.append(frame_data)
        self.total_frames += 1
        for label, count in counts.items():
            self.total_counts[label] += count

        return frame_data

    def _get_density(self, total_vehicles):
        for status, (low, high) in DENSITY_THRESHOLDS.items():
            if low <= total_vehicles < high:
                return status
        return 'HIGH'

    def get_rolling_average(self):
        """Average vehicle count over the history window."""
        if not self.frame_history:
            return 0
        return sum(f['total'] for f in self.frame_history) / len(self.frame_history)

    def get_dashboard_data(self):
        """
        Compile all metrics for the web dashboard.
        Called by the Flask app on each refresh.
        """
        recent = list(self.frame_history)[-10:] if self.frame_history else []
        current = self.frame_history[-1] if self.frame_history else {}

        return {
            'current_total':    current.get('total', 0),
            'current_density':  current.get('density', 'LOW'),
            'current_counts':   current.get('counts', {}),
            'rolling_average':  round(self.get_rolling_average(), 1),
            'session_total':    dict(self.total_counts),
            'frames_processed': self.total_frames,
            'session_duration': str(datetime.now() - self.session_start).split('.')[0],
            'history': [
                {'time': f['timestamp'], 'total': f['total'], 'density': f['density']}
                for f in recent
            ]
        }

    def reset(self):
        self.frame_history.clear()
        self.total_counts.clear()
        self.total_frames = 0
        self.session_start = datetime.now()
