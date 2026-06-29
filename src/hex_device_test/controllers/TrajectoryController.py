import time
import numpy as np

from enum import Enum

class TrajectoryStatus(Enum):
    IDLE            = 0
    Trajectory      = 1
    Return_Home     = 2 
    Finished        = 3
    Hold            = 4

DEFAULT_INIT_POS = [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]
DEFAULT_RETURN_HOME_DURATION = 10.0


class SegmentedTrajectoryPlanner:
    """Replay multiple trajectory segments sequentially with return-home pauses."""

    def __init__(
        self,
        waypoints,
        segment_ends,
        segment_duration=3.0,
        init_pos=None,
        hold_duration=300.0,
        return_home_duration=DEFAULT_RETURN_HOME_DURATION,
        interpolate=True,
    ):
        self.waypoints = waypoints
        self.segment_ends = list(segment_ends)
        self.segment_starts = [0] + self.segment_ends[:-1]
        self.segment_duration = segment_duration
        self.interpolate = interpolate
        self.init_pos = np.array(init_pos if init_pos is not None else DEFAULT_INIT_POS)
        self.hold_duration = hold_duration
        self.return_home_duration = return_home_duration

        self.current_segment_idx = 0
        self.loop_count = 0
        self.phase = TrajectoryStatus.Trajectory
        self.phase_start_time = None
        self._return_home_controller = None
        self.trajectory_started = False
        self.last_target_position = None
        self.current_waypoint_index = 0

    def start_trajectory(self):
        if not self.waypoints or not self.segment_ends:
            return False

        self.trajectory_started = True
        self.phase_start_time = time.time()
        self.current_segment_idx = 0
        self.loop_count = 0
        self.phase = TrajectoryStatus.Trajectory
        self._return_home_controller = None
        self.current_waypoint_index = self.segment_starts[0]
        return True

    def _segment_waypoint_range(self, segment_idx: int):
        start = self.segment_starts[segment_idx]
        end = self.segment_ends[segment_idx]
        return start, end

    def _segment_transition_count(self, segment_idx: int) -> int:
        start, end = self._segment_waypoint_range(segment_idx)
        return max(0, end - start - 1)

    def _skip_inter_segment_actions(self) -> bool:
        return self.hold_duration <= 0

    def _advance_after_segment(self):
        end_idx = self.segment_ends[self.current_segment_idx] - 1
        if self._skip_inter_segment_actions():
            print(
                f"[SegmentedTrajectory] segment {self.current_segment_idx + 1}/"
                f"{len(self.segment_ends)} finished at waypoint index {end_idx}, "
                f"continuing to next segment"
            )
            self._advance_after_hold()
            return

        end_pos = self.waypoints[end_idx]
        self._return_home_controller = ReturnHomeController(
            start_position=end_pos,
            home_position=self.init_pos,
            duration=self.return_home_duration,
        )
        self.phase = TrajectoryStatus.Return_Home
        self.phase_start_time = time.time()
        print(
            f"[SegmentedTrajectory] segment {self.current_segment_idx + 1}/"
            f"{len(self.segment_ends)} finished at waypoint index {end_idx}, returning to init_pos"
        )

    def _advance_after_return_home(self):
        self._return_home_controller = None
        if self.hold_duration <= 0:
            self._advance_after_hold()
            return

        self.phase = TrajectoryStatus.Hold
        self.phase_start_time = time.time()
        total = len(self.segment_ends)
        if self.current_segment_idx + 1 >= total:
            print(
                f"[SegmentedTrajectory] at init_pos, holding for {self.hold_duration}s "
                f"before next loop"
            )
        else:
            print(
                f"[SegmentedTrajectory] at init_pos, holding for {self.hold_duration}s "
                f"before segment {self.current_segment_idx + 2}/{total}"
            )

    def _advance_after_hold(self):
        self.current_segment_idx += 1
        if self.current_segment_idx >= len(self.segment_ends):
            self.loop_count += 1
            self.current_segment_idx = 0
            print(
                f"[SegmentedTrajectory] loop {self.loop_count} completed, "
                f"restarting from segment 1/{len(self.segment_ends)}"
            )
        else:
            print(
                f"[SegmentedTrajectory] starting segment "
                f"{self.current_segment_idx + 1}/{len(self.segment_ends)}"
            )

        self.phase = TrajectoryStatus.Trajectory
        self.phase_start_time = time.time()
        self.current_waypoint_index = self.segment_starts[self.current_segment_idx]

    def get_current_target(self):
        if not self.trajectory_started or not self.waypoints:
            return None

        if self.phase == TrajectoryStatus.Trajectory:
            return self._target_for_trajectory_phase()
        if self.phase == TrajectoryStatus.Return_Home:
            return self._target_for_return_home_phase()
        if self.phase == TrajectoryStatus.Hold:
            return self._target_for_hold_phase()
        if self.phase == TrajectoryStatus.Finished:
            self.last_target_position = self.init_pos
            return self.init_pos
        return None

    def _target_for_trajectory_phase(self):
        segment_idx = self.current_segment_idx
        start, end = self._segment_waypoint_range(segment_idx)
        segment_waypoints = self.waypoints[start:end]
        transition_count = self._segment_transition_count(segment_idx)
        local_elapsed = time.time() - self.phase_start_time

        if transition_count == 0:
            target = np.array(segment_waypoints[0])
            self.last_target_position = target
            self.current_waypoint_index = start
            self._advance_after_segment()
            return target

        sub_index = int(local_elapsed / self.segment_duration)
        if sub_index >= transition_count:
            target = np.array(segment_waypoints[-1])
            self.last_target_position = target
            self.current_waypoint_index = end - 1
            self._advance_after_segment()
            return target

        if not self.interpolate:
            target = np.array(segment_waypoints[sub_index])
            self.current_waypoint_index = start + sub_index
            self.last_target_position = target
            return target

        normalized_time = (local_elapsed % self.segment_duration) / self.segment_duration
        s = self._smooth_step(normalized_time)
        start_pos = np.array(segment_waypoints[sub_index])
        end_pos = np.array(segment_waypoints[sub_index + 1])
        target = start_pos + s * (end_pos - start_pos)
        self.current_waypoint_index = start + sub_index
        self.last_target_position = target
        return target

    def _target_for_return_home_phase(self):
        target, reached = self._return_home_controller.get_target_position()
        self.last_target_position = target
        if reached:
            self._advance_after_return_home()
        return target

    def _target_for_hold_phase(self):
        if time.time() - self.phase_start_time >= self.hold_duration:
            self._advance_after_hold()
        self.last_target_position = self.init_pos
        return self.init_pos

    def get_last_position(self):
        return self.last_target_position

    def _smooth_step(self, t):
        t = max(0.0, min(1.0, t))
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def get_current_segment_info(self):
        if not self.trajectory_started:
            return None

        info = {
            "segment_index": self.current_segment_idx,
            "segment_progress": 0.0,
            "total_elapsed": time.time() - self.phase_start_time,
            "phase": self.phase.name,
            "waypoint_index": self.current_waypoint_index,
            "loop_count": self.loop_count,
        }
        if self.phase == TrajectoryStatus.Trajectory:
            transition_count = self._segment_transition_count(self.current_segment_idx)
            if transition_count > 0:
                local_elapsed = time.time() - self.phase_start_time
                info["segment_progress"] = min(
                    1.0, (local_elapsed / self.segment_duration) / transition_count
                )
        elif self.phase == TrajectoryStatus.Hold and self.hold_duration > 0:
            info["segment_progress"] = min(
                1.0, (time.time() - self.phase_start_time) / self.hold_duration
            )
        return info


class TrajectoryPlanner:
    """Trajectory planner that supports smooth acceleration and deceleration planning"""
    
    def __init__(self, waypoints, segment_duration=3.0, interpolate=True):
        """
        Initialize trajectory planner
        waypoints: List of waypoints
        segment_duration: Duration of each trajectory segment (seconds)
        interpolate: If False, hold each waypoint as-is without S-curve blending
        """
        self.waypoints = waypoints
        self.segment_duration = segment_duration
        self.interpolate = interpolate
        
        self.current_waypoint_index = 0
        self.trajectory_started = False
        self.start_time = None
        self.last_target_position = None  # Store last commanded position
        
    def start_trajectory(self):
        """Start trajectory execution"""
        if not self.waypoints:
            return False
        
        self.trajectory_started = True
        self.start_time = time.time()
        self.current_waypoint_index = 0
        return True
        
    def get_current_target(self):
        """Get the target position at the current moment"""
        if not self.trajectory_started or not self.waypoints:
            return None
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        total_segments = len(self.waypoints)
        segment_index = int(elapsed_time / self.segment_duration) % total_segments
        
        segment_elapsed = elapsed_time % self.segment_duration
        normalized_time = segment_elapsed / self.segment_duration
        
        if not self.interpolate:
            target_position = np.array(self.waypoints[segment_index])
            self.current_waypoint_index = segment_index
            self.last_target_position = target_position
            return target_position

        start_waypoint = self.waypoints[segment_index]
        end_waypoint = self.waypoints[(segment_index + 1) % total_segments]
        
        # Use S-curve interpolation to calculate current position
        s = self._smooth_step(normalized_time)
        
        start_pos = np.array(start_waypoint)
        end_pos = np.array(end_waypoint)
        target_position = start_pos + s * (end_pos - start_pos)
        
        self.current_waypoint_index = segment_index
        self.last_target_position = target_position  # Store for potential return home
        
        return target_position
    
    def get_last_position(self):
        """Get the last commanded position"""
        return self.last_target_position
        
    def _smooth_step(self, t):
        """S-curve interpolation function that provides smooth acceleration and deceleration"""
        # Limit t to [0,1] range
        t = max(0.0, min(1.0, t))
        
        # Use 5th degree polynomial for smoother interpolation: 6t⁵ - 15t⁴ + 10t³
        return 6 * t**5 - 15 * t**4 + 10 * t**3
        
    def get_current_segment_info(self):
        """Get information about the current segment"""
        if not self.trajectory_started:
            return None
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        segment_index = int(elapsed_time / self.segment_duration) % len(self.waypoints)
        segment_elapsed = elapsed_time % self.segment_duration
        segment_progress = segment_elapsed / self.segment_duration
        
        return {
            'segment_index': segment_index,
            'segment_progress': segment_progress,
            'total_elapsed': elapsed_time
        }

class ReturnHomeController:
    """Controller for smooth return to home position"""
    
    def __init__(self, start_position, home_position, duration):
        """
        Initialize return home controller
        start_position: Starting position (current position when Ctrl+C is pressed)
        home_position: Target home position
        duration: Duration to reach home position (seconds)
        """
        self.start_position = np.array(start_position)
        self.home_position = np.array(home_position)
        self.duration = duration
        self.start_time = time.time()
        self.done = False
        
    def get_target_position(self):
        """Get the current target position during return home"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if np.allclose(self.start_position,self.home_position):
            self.done = True
        
        if elapsed_time >= self.duration or self.done:
            return self.home_position, True  # Reached home
        
        # if self.done:
        #     return self.home_position, True  # Reached home
        
        # Calculate normalized time [0, 1]
        t = elapsed_time / self.duration
        
        # Use S-curve interpolation for smooth motion
        s = self._smooth_step(t)
        
        # Interpolate between start and home position
        target_position = self.start_position + s * (self.home_position - self.start_position)
        
        return target_position, False  # Not yet reached home
    
    def _smooth_step(self, t):
        """S-curve interpolation function"""
        t = max(0.0, min(1.0, t))
        return 6 * t**5 - 15 * t**4 + 10 * t**3
