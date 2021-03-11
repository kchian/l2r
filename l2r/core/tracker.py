# ========================================================================= #
# Filename:                                                                 #
#    tracker.py                                                             #
#                                                                           #
# Description:                                                              #
#    Tracks the vehicle and evaluates metrics on the path taken             #
# ========================================================================= #

import time

import numpy as np

from envs.utils import GeoLocation

# Assumed max progression, in number of indicies, in one RacingEnv.step()
MAX_PROGRESSION = 100

# Early termination for very poor performance
PROGRESS_THRESHOLD = 100
CHECK_PROGRESS_AT = 299

# constants
MPS_TO_KPH = 3.6
EPSILON = 1e-6

class ProgressTracker(object):
	"""Progress tracker for the vehicle. This class serves to track the number
	of laps that the vehicle has completed and how long each one took. It also
	evaluates for termination conditions.

	:param n_indices: number of indicies of the centerline of the track
	:type n_indices: int
	:param inner_track: path of the inner track boundary
	:type inner_track: matplotlib.path
	:param outer_track: path of the outer track boundary
	:type outer_track: matplotlib.path
	:param centerline: path of the centerline
	:type centerline: np array
	:param car_dims: dimensions of the vehicle in meters [len, width]
	:type car_dims: list of floats
	:param obs_delay: time delay between a RacingEnv action and observation
	:type obs_delay: float
	:param max_timesteps: maximum number of timesteps in an episode
	:type max_timesteps: int
	:param not_moving_ct: maximum number of vehicle can be stationary before
	  being considered stuck
	:type not_moving_ct: int
	:param debug: debugging print statement flag
	:type debug: boolean
	"""
	def __init__(self, n_indices, inner_track, outer_track, centerline,
		         car_dims, obs_delay, max_timesteps, not_moving_ct,
		         debug=False):
		self.n_indices = n_indices
		self.halfway_idx = n_indices//2
		self.inner_track = inner_track
		self.outer_track = outer_track
		self.centerline = centerline
		self.car_dims = car_dims
		self.obs_delay = obs_delay
		self.max_timesteps = max_timesteps
		self.not_moving_ct = not_moving_ct
		self.debug = debug
		self.reset(None)

	def reset(self, start_idx):
		"""Reset the tracker for the next episode.

		:param start_idx: index on the track's centerline which the vehicle is
		  nearest to
		:type start_idx: int
		"""
		self.start_idx = start_idx
		self.last_idx = start_idx
		self.lap_start = None
		self.last_update_time = None
		self.lap_times = []
		self.halfway_flag = False
		self.ep_step_ct = 0
		self.transitions = []

	def update(self, idx, d, e, n, u, yaw, bp):
		"""Update the tracker based on current position. The tracker also keeps
		track of the yaw, brake pressure, centerline displacement, and
		calculates the offical metrics for the environment.

		:param idx: index on the track's centerline which the vehicle is
		  nearest to
		:type idx: int
		:param d: displacement from centerline, defined at distance to the
		  nearest point on the centerline
		:type d: float
		:param e: east coordinate
		:type e: float
		:param n: north coordinate
		:type n: float
		:param u: up coordinate
		:type u: float
		:param yaw: vehicle heading, in radians
		:type yaw: float
		:param bp: brake pressure, per wheel
		:type bp: array of shape (4,)
		"""
		now = time.time()

		if self.lap_start is None:
			self.start_time = now
			self.last_update_time = now
			self.lap_start = now - self.obs_delay
			return

		if idx >= self.n_indices:
			raise Exception('Index out of bounds')

		# shift idx such that the start index is at 0
		idx -= self.start_idx
		idx += self.n_indices if idx < 0 else 0

		# validate vehicle isn't just oscillating near the starting point
		if self.last_idx <= self.halfway_idx and idx >= self.halfway_idx:
			self.halfway_flag = True

		n_out = self._count_wheels_oob(e, n, yaw)
		self._store(e, n, u, idx, yaw, d, bp, n_out)

		# set halfway flag, if necessary
		_ = self.check_lap_completion(idx, now)
		self.ep_step_ct += 1
		self.last_update_time = now
		self.last_idx = idx

	def _store(self, e, n, u, idx, yaw, d, bp, n_out):
		"""Transitions are stored as a list of lists

		:param e: east coordinate
		:type e: float
		:param n: north coordinate
		:type n: float
		:param u: up coordinate
		:type u: float
		:param idx: nearest centerline index to current position
		:type idx: int
		:param d: distance to centerline[idx]
		:type d: float
		:param bp: brake pressure, per wheel
		:type bp: array of shape (4,)
		:param n_out: number of wheels out-of-bounds
		:type n_out: int
		"""
		b = np.average(bp)
		self.transitions.append([e, n, u, idx, d, yaw, b, n_out])

	def check_lap_completion(self, shifted_idx, now):
		"""Check if we completed a lap. To prevent vehicles from oscillating
		near the start line, the vehicle must first cross the halfway mark of
		the track. If the vehicle has cross the halfway mark, we then check if
		the vehicle has crossed the index that it started at. If the vehicle
		crosses multiple indicies in one time step when this happens, the lap's
		end time is a linear interpolation between the time of the prior 
		update and the current time

		:param shifted_idx: index on the track's centerline which the vehicle
		  is nearest to shifted based on the starting index on the track
		:type shifted_idx: int
		:param now: time of the update
		:type now: float
		"""
		if not self.halfway_flag:
			return False

		if shifted_idx < MAX_PROGRESSION:
			ct = shifted_idx + (self.n_indices - self.last_idx)
			lap_end = now - (now - self.last_update_time) / ct * shifted_idx
			lap_time = round(lap_end - self.lap_start, 2)
			self.lap_times.append(lap_time)
			self.lap_start = lap_end
			self.halfway_flag = False
			return True

		return False

	def is_complete(self):
		"""Determine if the episode is complete due to finishing 3 laps,
		remaining in the same position for too long (stuck), exceeding the
		maximum number of timestepsor, or going out-of-bounds. If all 3 laps
		were successfully completed, the total time is also returned.

		:return: complete, info which includes metrics if successful
		:rtype: boolean, str, list of floats, float
		"""
		info = self._is_terminal()

		if info['stuck'] or info['not_progressing'] or info['dnf'] or info['oob']:
			return True, self.append_metrics(info) # info

		if info['success']:
			return True, self.append_metrics(info)

		return False, info

	def append_metrics(self, info):
		"""Environment metrics
		"""
		transitions = np.asarray(self.transitions).T
		path = transitions[0:2]
		total_distance = ProgressTracker._path_length(path)
		total_time = self.last_update_time - self.start_time
		avg_speed = total_distance / total_time * MPS_TO_KPH
		avg_cline_displacement = np.average(transitions[4])
		avg_curvature = ProgressTracker._path_curvature(path)
		track_curvature = ProgressTracker._path_curvature(self.centerline.T)
		brake_pressure = transitions[-2]

		info['total_distance'] = round(total_distance, 2)
		info['total_time'] = round(total_time, 2)
		info['average_speed_kph'] = round(avg_speed, 2)
		info['average_centerline_displacement'] = round(avg_cline_displacement, 2)
		info['average_curvature'] = avg_curvature
		info['trajectory_smoothness'] = round(track_curvature / avg_curvature, 3)
		info['n_unsafe_steps'] = np.sum(transitions[-1])
		return info

	@staticmethod
	def _path_length(path):
		"""Calculate length of a path
		"""
		x, y = path[0], path[1]
		_x = np.square(x[:-1]-x[1:])
		_y = np.square(y[:-1]-y[1:])
		return np.sum(np.sqrt(_x + _y))

	@staticmethod
	def _path_curvature(path):
		"""Returns the average absolute curvature (numberical estimate) of the
		path
		"""
		x, y = path[0], path[1]

		dx = 0.5*(x[2:]-x[:-2])
		dy = 0.5*(y[2:]-y[:-2])
		d2x = x[2:]-2*x[1:-1] + x[0:-2]
		d2y = y[2:]-2*y[1:-1] + y[0:-2]
		k = (dx*d2y-dy*d2x) / (np.square(dx)+np.square(dy)+EPSILON)**(3.0/2.0)

		return np.average(np.abs(k))

	def _is_terminal(self):
		"""Determine if the environment is in a terminal state
		"""
		info = {
			'stuck': False,
			'oob': False,
			'success': False,
			'dnf': False,
			'not_progressing': False,
			'lap_times': self.lap_times
		}

		if len(self.lap_times) >= 3:
			info['success'] = True
			info['total_time'] = round(sum(self.lap_times), 2)
			info['pct_complete'] = 100.00

		total_idxs = self.last_idx + self.n_indices*len(self.lap_times)
		info['pct_complete'] = round(100*total_idxs/(3*self.n_indices), 1)

		if len(self.transitions) > self.not_moving_ct:
			if self.transitions[-1] == self.transitions[-self.not_moving_ct]:
				info['stuck'] = True

		if self.ep_step_ct == CHECK_PROGRESS_AT and total_idxs < PROGRESS_THRESHOLD:
			info['not_progressing'] = True

		if self.ep_step_ct >= self.max_timesteps:
			info['dnf'] = True

		if self._car_out_of_bounds():
			info['oob'] = True

		return info

	def _car_out_of_bounds(self):
		"""At most 1 wheel can be outside of the driveable area

		:return: True if vehicle out-of-bounds
		:rtype: bool
		"""
		return len(self.transitions) > 0 and self.transitions[-1][-1] > 1

	def _count_wheels_oob(self, e, n, yaw):
		"""Count number of wheels that are out of the drivable area

		:return: Number of wheels out of drivable area
		:rtype: int
		"""
		car_corners = GeoLocation.get_corners((e,n), yaw, self.car_dims)

		# At most 1 wheel can be out-of-track
		n_out_inside = np.count_nonzero(self.inner_track.contains_points(car_corners))
		if n_out_inside > 0:
			return n_out_inside

		return 4 - np.count_nonzero(self.outer_track.contains_points(car_corners))
