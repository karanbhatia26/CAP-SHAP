import numpy as np
from typing import List, Tuple, Optional

class PhaseDetector:
    def __init__(
        self,
        method: str = "reward_changepoint",
        threshold: float = 0.5,
        min_len: int = 10,
        fixed_num_phases: int = 3,
        smooth_window: int = 5,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.min_len = max(1, int(min_len))
        self.fixed_num_phases = max(1, int(fixed_num_phases))
        self.smooth_window = max(1, int(smooth_window))

    def segment_episode(
        self,
        rewards_env: np.ndarray,
    ) -> List[Tuple[int, int]]:
        r = rewards_env.squeeze()
        T = int(r.shape[0])
        if T == 0:
            return []

        if self.method == "fixed":
            return self._fixed_segments(T)
        else:
            segs = self._changepoint_segments(r)
            if len(segs) == 0:
                return self._fixed_segments(T)
            return segs

    def segment(self, rewards_env: np.ndarray) -> List[Tuple[int, int]]:
        """Alias for segment_episode for consistency"""
        return self.segment_episode(rewards_env)

    def _fixed_segments(self, T: int) -> List[Tuple[int, int]]:
        P = min(self.fixed_num_phases, T // self.min_len if T >= self.min_len else 1)
        if P <= 0:
            P = 1
        base = T // P
        rem = T % P
        segs = []
        start = 0
        for p in range(P):
            length = base + (1 if p < rem else 0)
            end = start + max(length, 1)
            segs.append((start, end))
            start = end
        segs = self._enforce_min_len(segs, T)
        return segs

    def _changepoint_segments(self, r: np.ndarray) -> List[Tuple[int, int]]:
        T = r.shape[0]
        if self.smooth_window > 1 and T > 1:
            w = self.smooth_window
            pad = min(w - 1, T - 1)
            rp = np.pad(r, (pad, 0), mode="edge")
            kernel = np.ones(w) / w
            rs = np.convolve(rp, kernel, mode="valid")[:T]
        else:
            rs = r
        diffs = np.abs(np.diff(rs, prepend=rs[0]))
        scale = np.std(rs) + 1e-6
        idxs = np.where(diffs > self.threshold * scale)[0]
        cut_points = [0]
        last = 0
        for i in idxs:
            if i - last >= self.min_len:
                cut_points.append(i)
                last = i
        if T - last >= self.min_len:
            cut_points.append(T)
        else:
            if cut_points[-1] != T:
                cut_points[-1] = T
        segs = [(cut_points[k], cut_points[k+1]) for k in range(len(cut_points)-1)]
        segs = self._enforce_min_len(segs, T)
        return segs

    def _enforce_min_len(self, segs: List[Tuple[int, int]], T: int) -> List[Tuple[int, int]]:
        if not segs:
            return [(0, T)]
        out: List[Tuple[int, int]] = []
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if (cur_e - cur_s) < self.min_len:
                cur_e = e
            else:
                out.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        out.append((cur_s, cur_e))
        out = [(max(0, s), min(T, e)) for s, e in out if e > s]
        if not out:
            out = [(0, T)]
        return out

# Alias for compatibility
PhaseSegmenter = PhaseDetector
