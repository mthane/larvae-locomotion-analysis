# ==============================================================================
# Track class
# ==============================================================================

import numpy as np
import os

import configparser as ConfigParser

from scipy.signal import argrelmax
from MiscFunctions import MiscFunctions as mf
import pickle as cPickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from matplotlib.patches import Wedge

from matplotlib.patches import Polygon

from matplotlib.ticker import MaxNLocator

degree_sign ='degree'
font_size = 12

class Track:
    '''class for a single track, and respective figures'''

    def __init__(self, par=None, experiment=None,
                 condition=None,
                 trial=None, track_number=None):

        file_path = ''
        self.par = par
	
        if (
                experiment is not None and
                condition is not None and
                trial is not None):
            file_path = '/'.join((par['parent_dir'], experiment,
                                  par['tracked_on'],
                                  par['group_name'], condition,
                                  trial, str(track_number) + '.csv'))
            print(file_path)
            dir_trial = '/'.join((par['parent_dir'], experiment,
                                  par['tracked_on'],
                                  par['group_name'], condition,
                                  trial))

        if (file_path == ''):
            data_ok = False
            print('       data_init error: File path could not be determined')
	
        print(file_path)

        # if file exists
        if os.path.isfile(file_path):

            print('\n   - trial:', trial)
            # Check for metadata.txt
            trial_metadata = os.path.join(dir_trial, 'vidAndLogs/metadata.txt')
            if os.path.isfile(trial_metadata):
                trial_config = ConfigParser.RawConfigParser()
                trial_config.read(trial_metadata)

            try:
                par['odor_A'] = np.array(
                    list(map(float,
                             trial_config.get(
                                 'Trial Data', 'OdorALocation').split(','))))
                print("Setting odor position for trial: " + str(par['odor_A']))
            except:
                print("ERROR: Problem setting odor pos for trial " + str(trial))
                par['odor_A'] = [0, 0]

            try:
                par['odor_B'] = np.array(
                    list(map(float,
                             trial_config.get(
                                 'Trial Data', 'OdorBLocation').split(','))))
            except:
                par['odor_B'] = [0, 0]
            # load csv data
            data_init = np.genfromtxt(file_path, delimiter=',')

            # check data
            data_ok = True
            # if par['odor_A'] is None:
            #    data_ok = False
            if data_init.ndim == 2:
		
                # size tests
                if (data_init.shape[1] != 78):
                    data_ok = False
                    print('       # data_init error: wrong column number')

                # check for valid duration
                if (np.sum(data_init[:, 77] == 0) <
                        par['minimal_duration'] / float(par['dt'])):
                    print('       # data_init error: track too short')
                    data_ok = False

            else:
                data_ok = False
                print('       # data_init error: ndim wrong!')

            print("Data ok: " + str(data_ok))
            # if data_init is ok
            if data_ok:

                # init
                self.experiment = experiment
                self.condition = condition
                self.trial = trial
                self.track_number = track_number
                self.ok = True

                # Transform frames to sec:
                data_init[:, 0] = data_init[:, 0] / float(par['fps'])

                # init valid_frame_tmp
                valid_frame_tmp = np.zeros(data_init.shape[0]) * np.nan
                valid_frame_tmp[data_init[:, 77] == 0] = 1.

                # rotate all data so that the odor is placed on the top
                #  middle
                print("Rotating based on odor_A: " + str(par['odor_A']))
                print("Example value before: " + str(data_init[:, 1:3][0]))

                rotation_angle = mf.clockwise_angle_from_first_to_second_vector(
                    par['odor_A'],
                    np.array([0.0, np.linalg.norm(par['odor_A'])])
                )
                for col in range(1, 71, 2):
                    data_init[:, col: col + 2] = np.array(
                        mf.rotate_vector_clockwise(
                            rotation_angle,
                            [data_init[:, col],
                             data_init[:, col + 1]])).T
                par['odor_A'] = [0.0, np.linalg.norm(par['odor_A'])]
                print("Changed odor to: " + str(par['odor_A']))
                print("Example value after: " + str(data_init[:, 1:3][0]))
                # convolve data_init
                for col in range(1, 72):
                    data_init[:, col] = np.convolve(
                        data_init[:, col],
                        np.ones((2 * par['n_filter_points'] + 1)) /
                        float(2 * par['n_filter_points'] + 1),
                        mode='same'
                    )

                # convolve valid_frame_tmp by the same filter
                valid_frame_tmp = np.convolve(
                    valid_frame_tmp,
                    np.ones((2 * par['n_filter_points'] + 1)) /
                    float(2 * par['n_filter_points'] + 1),
                    mode='same'
                )

                # correct row number for all columns and valid_frame_tmp
                if par['n_filter_points'] > 0:
                    data_init = data_init[par['n_filter_points']: -
                    par['n_filter_points'], :]
                    valid_frame_tmp = valid_frame_tmp[par['n_filter_points']: -
                    par['n_filter_points']]

                # time
                self.time = data_init[:, 0]

                # absolute spine points
                self.spine = []
                for idx in range(12):
                    self.spine.append(data_init[:, 1 + 2 * idx: 3 + 2 * idx])

                # absolute contour points
                self.contour = []
                for idx in range(22):
                    self.contour.append(
                        data_init[:, 25 + 2 * idx: 27 + 2 * idx])

                # centroid 69, 70
                self.centroid = data_init[:, 69: 71]

                # spine vectors
                self.spine_vectors = []
                for idx in range(11):
                    self.spine_vectors.append(self.spine[idx + 1] -
                                              self.spine[idx])

                # valid_frame (=1. if valid, =np.nan else)
                self.valid_frame = np.ones(len(self.time))
                self.valid_frame[np.isnan(valid_frame_tmp)] = np.nan

                # cut data to the desired time frame
                # if par['start_time'] != 0 or
                #    par['end_time'] != par['duration']:
                if par['time_restrict'] is True:
                    self.cut_track_before_analysis(
                        par,
                        par['start_time'],
                        par['end_time'])

                self.check_preliminary_track(par)
                if not self.ok:
                    print('       (data_init is not ok)')

            # if data_init is not ok
            else:
                print('       (data_init is not ok)')
                self.ok = False

        # if file path does not exist
        else:
            print('       (file_path is not valid: ' + str(file_path) + ')')
            self.ok = False

    #def __new__(arg):
    #    print("Calling __new__ with arg: " + str(arg))
    def write_database(par,
                       experiment,
                       group,
                       conditions,
                       which_trials,
                       which_tracks):
        '''analyze and save tracks as pkl-files for all csv-files
           specified by arguments'''

        print('\n' + 50 * '=' + '\n\nWriting database...')

        # dir of experiment
        dir_experiment = (par['parent_dir'] + '/' + experiment + '/' +
                          par['tracked_on'])

        # check if dir
        if os.path.isdir(dir_experiment):
            print('\n- experiment:', experiment)
        else:
            print('Error:', dir_experiment, 'is not a directory!')

        # for all conditions
        for condition in conditions:

            # dir_condition
            dir_condition = dir_experiment + '/' + group + '/' + condition
            full_condition = group + '_-_' + condition
            # check if dir
            if os.path.isdir(dir_condition):
                print('\n - dir: ', dir_condition)
                print('\n - group: ', group)
                print('\n - condition:', full_condition)
            else:
                print('Error:', dir_condition, 'is not a directory!')

            # for which trials?
            if which_trials == 'all':
                trials = os.listdir(dir_condition)
            if which_trials == 'first':
                trials = [os.listdir(dir_condition)[0]]
            if isinstance(which_trials, list):
                trials = np.copy(which_trials)

            # for all trials
            for trial in trials:

                # real trial_id
                # trial_id = trial

                # check if dir
                dir_trial = '/'.join((dir_condition, trial))

                if os.path.isdir(dir_trial):

                    print('\n   - trial:', trial)
                    print('\n   - trial dir:', dir_trial)

                    # delete all pkl files first
                    if True:
                        pkl_files = [x for x in os.listdir(dir_trial)
                                     if par['path_suffix'] in x]
                        for pkl_file in pkl_files:
                            os.remove('/'.join((dir_trial, pkl_file)))

                    all_track_names = [x for x in os.listdir(dir_trial)
                                       if '.csv' in x]

                    # for which tracks?
                    if which_tracks == 'all':
                        track_names = all_track_names
                    if which_tracks == 'first':
                        track_names = [all_track_names[0]]
                    if isinstance(which_tracks, list):
                        track_names = [str(track_number) + '.csv'
                                       for track_number in which_tracks]

                    # for all track.csv
                    for track_name in track_names[1:]:

                        # check if file
                        file_path = '/'.join((dir_trial, track_name))

                        if os.path.isfile(file_path):

                            print('     - track:', track_name)

                            # init track
                            track = Track(par, experiment=experiment,
                                          condition=condition,
                                          trial=trial,
                                          track_number=int(track_name[:-4]))


                            # if track is ok
                            if track.ok:
                                # analyze and save track
                                track.compute_time_series_variables(par)
                                track.detect_steps(par)
                                track.detect_HCs(par)
                                track.compute_scalar_variables(par)
                                track.check_analyzed_track(par)
                                # track.figure_track_on_dish(
                                #    par, ['start', 'end'],
                                #    'single_track' + str(track.trial) +
                                #    str(track.track_number))
                                
                                track.save(par)
                            
                                # track.close()

        print('\n...done')


    def cut_track_before_analysis(self, par, t_start, t_end):
        ''''has to be run before the analysis'''

        idx_start = np.argmin(np.abs(self.time - t_start))
        idx_end = np.argmin(np.abs(self.time - t_end))

        for key in ['time', 'valid_frame']:
            setattr(self, key, getattr(self, key)[idx_start: idx_end])

        for key in ['spine', 'spine_vectors', 'contour']:
            for j in range(len(getattr(self, key))):
                self.__dict__[key][j] = \
                    self.__dict__[key][j][idx_start: idx_end]

    def compute_time_series_variables(self, par):
        '''computes equal-length time-series variables'''

        if par['odor_A'] is not None:
            # distance to odorA from head
            self.distance = np.sqrt(np.sum((
                                                   par['odor_A'] - self.spine[11]) ** 2, 1))

        # midpoint speed
        # self.midpoint_speed = 1./float(par['dt']) * np.abs(np.diff(
        #     self.spine[5], axis=0))
        self.midpoint_speed = 1. / float(par['dt']) * np.sqrt(
            np.sum(np.diff(self.spine[5], axis=0) ** 2, 1))
        # self.midpoint_speed = np.insert(self.midpoint_speed, 0, 0)
        # correct length
        self.midpoint_speed = np.hstack(
            [self.midpoint_speed, np.nan])

        # centroid speed
        # self.centroid_speed = 1./float(par['dt']) * np.abs(np.diff(
        #     self.centroid, axis=0))
        self.centroid_speed = 1. / float(par['dt']) * np.sqrt(
            np.sum(np.diff(self.centroid, axis=0) ** 2, 1))
        # self.centroid_speed = np.insert(self.centroid_speed, 0, 0)
        # correct length
        self.centroid_speed = np.hstack(
            [self.centroid_speed, np.nan])

        # back_vector, normalized, from spine point 1 to 5
        self.back_vector = self.spine[5] - self.spine[1]
        self.back_vector /= np.sqrt(np.sum(
            self.back_vector ** 2, 1)).reshape(len(self.time), 1)

        # back_vector_angular_speed
        self.back_vector_angular_speed = 1. / float(par['dt']) * np.sqrt(
            np.sum(np.diff(self.back_vector, axis=0) ** 2, 1))

        # correct length
        self.back_vector_angular_speed = np.hstack(
            [self.back_vector_angular_speed, np.nan])
        # head_vector_angular_speed
        head_vector = self.spine[10] - self.spine[8]
        self.head_vector = head_vector
        self.head_vector_angular_speed = (
                1. / float(par['dt']) * mf.clockwise_angle_from_first_to_second_vector(
            [head_vector[:-1, i] for i in [0, 1]],
            [head_vector[1:, i] for i in [0, 1]]))

        # convolve head_vector_angular_speed by 0.45 seconds
        n_points = np.round(0.45 / float(par['dt']))
        # print n_points
        self.head_vector_angular_speed = np.convolve(
            np.ones(int(n_points)) / n_points,
            self.head_vector_angular_speed, mode='same')

        # correct length
        self.head_vector_angular_speed = np.hstack(
            [self.head_vector_angular_speed, np.nan])

        # convolve valid_frame by broadest used temporal filter of 0.45
        # seconds too
        tmp = np.convolve(np.ones(int(n_points)) / n_points,
                          self.valid_frame, mode='same')
        self.valid_frame[np.isnan(tmp)] = np.nan

        # set last to nan
        self.valid_frame[-1] = np.nan

        # bending angle, angle between back and front vector
        # NOTE: Multiplied by (-1) to match Barcelona manuscript
        self.bending_angle = -1.0 * mf.clockwise_angle_from_first_to_second_vector(
            [self.back_vector[:, i] for i in [0, 1]],
            [(self.spine[10] - self.spine[5])[:, i] for i in [0, 1]]
        )

        # convolve bending angle by 0.3 seconds
        n_points = np.round(0.3 / float(par['dt']))
        self.bending_angle = np.convolve(
            self.bending_angle, np.ones(int(n_points)) / n_points, mode='same')

        if (par['odor_A'] != None):
            # bearing angle
            # MANOS: NOTE: Discrepancy between paper and actual measurement
            #              The second vector goes from spine point 6 to odor
            #              NOT from spine point 2 to odor as in the paper!!!
            self.bearing_angle = mf.clockwise_angle_from_first_to_second_vector(
                [self.back_vector[:, i] for i in [0, 1]],
                [(par['odor_A'] - self.spine[5])[:, i] for i in [0, 1]]
            )

            # heading angle
            self.heading_angle = mf.clockwise_angle_from_first_to_second_vector(
                [self.head_vector[:, i] for i in [0, 1]],
                [(par['odor_A'] - self.spine[8])[:, i] for i in [0, 1]]
            )

        # spine length
        self.spine_length = np.sum(
            [np.sqrt(np.sum(self.spine_vectors[idx] ** 2, 1))
             for idx in range(11)], 0)

        # tail_speed parallel to back vector (= forward direction)
        self.tail_speed_forward = 1. / float(par['dt']) * np.sum(
            self.back_vector[:-1, :] *
            np.diff(self.spine[0], axis=0), 1)

        # convolve tail_speed_forward by 0.3 seconds
        n_points = np.round(0.3 / float(par['dt']))
        self.tail_speed_forward = np.convolve(self.tail_speed_forward,
                                              np.ones(int(n_points)) / n_points,
                                              mode='same')

        # correct length
        self.tail_speed_forward = np.hstack([self.tail_speed_forward, np.nan])

        # head_speed parallel to forward vector
        self.head_speed_forward = 1. / float(par['dt']) * np.sum(
            self.back_vector[:-1, :] * np.diff(self.spine[11], axis=0), 1)

        # convolve head_speed_forward by 0.3 seconds
        n_points = np.round(0.3 / float(par['dt']))
        self.head_speed_forward = np.convolve(self.head_speed_forward,
                                              np.ones(int(n_points)) / n_points,
                                              mode='same')

        # correct length
        self.head_speed_forward = np.hstack([self.head_speed_forward, np.nan])

        # spine angles
        if False:
            self.spine_angles = []
            for idx in range(10):
                self.spine_angles.append(
                    clockwise_angle_from_first_to_second_vector(
                        [self.spine_vectors[idx][:, i] for i in [0, 1]],
                        [self.spine_vectors[idx + 1][:, i] for i in [0, 1]]
                    ))

    def detect_steps(self, par):
        '''detects tail point forward steps'''

        # all idx of forward movements
        # consider 0.3 seconds to left and right for comparison
        n_points = int(np.round(0.3 / float(par['dt'])))
        # this produces a Warning because of nans
        # print "Tail Speed Forward: \n" + str(self.tail_speed_forward)
        step_idx_all = argrelmax(self.tail_speed_forward[0:-1],
                                 order=n_points)[0]

        # step idx (all idxs of true steps, including the last one)
        self.step_idx = step_idx_all[
            self.tail_speed_forward[step_idx_all] >=
            par['threshold_tail_speed_forward']]

        # false step idx (this is just used for some plots on false steps)
        self.step_idx_false = step_idx_all[
            self.tail_speed_forward[step_idx_all] <
            par['threshold_tail_speed_forward']]

        # check for step_idx which current step has invalid next step
        invalid_next_step = np.zeros(len(self.step_idx)).astype(bool)
        for step_count in range(len(self.step_idx) - 1):
            if np.isnan(self.valid_frame[self.step_idx[step_count]:
            self.step_idx[step_count + 1]]).any():
                invalid_next_step[step_count] = True

        # inter-step-interval (same length as step_idx)
        self.INS_interval = np.diff(self.time[self.step_idx])
        self.INS_interval = np.hstack([self.INS_interval, np.nan])
        self.INS_interval[invalid_next_step] = np.nan

        # inter-step-distance (same length as step_idx)
        self.INS_distance = np.sqrt(np.sum(np.diff(
            self.spine[0][self.step_idx, :], axis=0) ** 2, 1))
        self.INS_distance = np.hstack([self.INS_distance, np.nan])
        self.INS_distance[invalid_next_step] = np.nan

        # inter-step-turn (same length as step_idx)
        # NOTE: Multiplied by (-1) to match Barcelona manuscript
        self.INS_turn = mf.clockwise_angle_from_first_to_second_vector(
            [self.back_vector[self.step_idx][:-1, i] for i in [0, 1]],
            [self.back_vector[self.step_idx][1:, i] for i in [0, 1]]
        ) * (-1)
        self.INS_turn = np.hstack([self.INS_turn, np.nan])
        self.INS_turn[invalid_next_step] = np.nan

    def detect_HCs(self, par):
        '''detects lateral active head casts'''

        # get all start and end idx
        above_threshold = np.zeros(len(self.time))

        above_threshold[self.valid_frame == 1.] = (
                np.abs(self.head_vector_angular_speed[self.valid_frame == 1.]) >
                par['threshold_head_vector_angular_speed']).astype(float)

        idx, = np.diff(above_threshold).nonzero()
        idx += 1
        # if starts with head casting
        if above_threshold[0] == 1:
            idx = np.r_[0, idx]
        # if ends with head casting
        if above_threshold[-1] == 1:
            idx = np.r_[idx, above_threshold.size]
        idx.shape = (-1, 2)
        HC_start_idx_all = idx[:, 0]
        HC_end_idx_all = idx[:, 1]

        # check if frames are valid during HC
        valid_HC_idx = np.ones(len(HC_start_idx_all)).astype(bool)
        for HC_count in range(len(HC_start_idx_all)):
            if np.isnan(self.valid_frame[HC_start_idx_all[HC_count]:
            HC_end_idx_all[HC_count]]).any():
                valid_HC_idx[HC_count] = False
        HC_start_idx_all = HC_start_idx_all[valid_HC_idx]
        HC_end_idx_all = HC_end_idx_all[valid_HC_idx]

        # init
        n_steps_in_HC_all = np.zeros(len(HC_start_idx_all))
        max_back_vector_angular_speed_all = np.zeros(len(HC_start_idx_all))

        # for all idx
        if len(HC_start_idx_all) > 0:
            for HC_idx, (start_idx, end_idx) in enumerate(zip(HC_start_idx_all,
                                                              HC_end_idx_all)):
                max_back_vector_angular_speed_all[HC_idx] = np.max(
                    self.back_vector_angular_speed[start_idx: end_idx])

                n_steps_in_HC_all[HC_idx] = np.sum(
                    (self.step_idx >= start_idx) *
                    (self.step_idx <= end_idx))

        # selection criteria for true HC
        HC_true = ((n_steps_in_HC_all <= par['threshold_n_steps_per_HC']) *
                   (max_back_vector_angular_speed_all <
                    par['threshold_back_vector_angular_speed']))

        self.HC_start_idx = HC_start_idx_all[HC_true]
        self.HC_end_idx = HC_end_idx_all[HC_true]

        self.max_back_vector_angular_speed = (
            max_back_vector_angular_speed_all[HC_true])
        self.max_back_vector_angular_speed_false = (
            max_back_vector_angular_speed_all[~HC_true])

        self.n_steps_in_HC = n_steps_in_HC_all[HC_true]
        self.n_steps_in_HC_false = n_steps_in_HC_all[~HC_true]

        # HC_angle
        self.HC_angle = (
                self.bending_angle[self.HC_end_idx] -
                self.bending_angle[self.HC_start_idx])

        self.HC_angle_false = (self.bending_angle[HC_end_idx_all[~HC_true]] -
                               self.bending_angle[HC_start_idx_all[~HC_true]])

        # HC direction
        self.HC_direction = (
            np.sign(self.head_vector_angular_speed[self.HC_start_idx]))

        # HC initiation (=time-series variable)
        self.HC_initiation = np.zeros(len(self.time))
        self.HC_initiation[self.HC_start_idx] = 1.
        self.HC_initiation[np.isnan(self.head_vector_angular_speed)] = np.nan

        # HC or run
        self.HC = np.zeros(len(self.time))
        HC_ranges = list(zip(self.HC_start_idx, self.HC_end_idx + 1))
        HC_indices = [
            y for sublist in [list(range(*x)) for x in HC_ranges] for y in sublist]
        self.HC[HC_indices] = 1
        # next event after current step is step
        self.next_event_is_step = np.zeros(len(self.step_idx)).astype(bool)
        HC_idxs = np.hstack([self.HC_start_idx, self.HC_end_idx])

        # for all steps
        for step_count in range(len(self.step_idx) - 1):
            if (
                    # no HC_idxs between steps
                    (np.sum((self.step_idx[step_count] <= HC_idxs) *
                            (self.step_idx[step_count + 1] >= HC_idxs)) == 0) and
                    # no nans between steps
                    (~np.isnan(self.valid_frame[
                               self.step_idx[step_count]:
                               self.step_idx[step_count + 1]])).all()
            ):
                self.next_event_is_step[step_count] = True

        # step-HC and HC-step interval for current step
        self.step_HC_interval = np.zeros(len(self.HC_start_idx)) * np.nan
        self.HC_step_interval = np.zeros(len(self.HC_start_idx)) * np.nan

        # for all HCs
        for HC_count in range(len(self.HC_start_idx)):

            # get last and next steps
            last_step_idxs = (
                self.step_idx[self.step_idx < self.HC_start_idx[HC_count]])
            next_step_idxs = (
                self.step_idx[self.step_idx > self.HC_start_idx[HC_count]])

            if len(last_step_idxs) > 0 and len(next_step_idxs) > 0:

                last_step_idx = last_step_idxs[-1]
                next_step_idx = next_step_idxs[0]

                if (~np.isnan(self.valid_frame[last_step_idx:
                next_step_idx])).all():
                    self.step_HC_interval[HC_count] = (
                            self.time[self.HC_start_idx[HC_count]] -
                            self.time[last_step_idx])
                    self.HC_step_interval[HC_count] = (
                            self.time[next_step_idx] -
                            self.time[self.HC_start_idx[HC_count]])

    def compute_scalar_variables(self, par):
        '''computes some additional scalar variables'''

        # duration
        self.duration = self.time[-1] - self.time[0]

        # mean spine length
        self.mean_spine_length = np.nanmean(self.spine_length)

        # mean inter step interval (without pairs with HCs)
        self.mean_INS_interval = np.nanmean(
            self.INS_interval[self.next_event_is_step])

        # mean inter step distance (without pairs with HCs)
        self.mean_INS_distance = np.nanmean(
            self.INS_distance[self.next_event_is_step])

    def check_preliminary_track(self, par):
        '''checks preliminary track'''
        if len(self.time) < par['minimal_duration'] * float(par['fps']):
            self.ok = False
            # there may be other issues,and bugs,add more tests here if you like

    def check_analyzed_track(self, par):
        '''checks analyzed track'''

        if not ((
                        len(self.step_idx) ==
                        len(self.INS_turn) ==
                        len(self.INS_interval) ==
                        len(self.INS_distance) ==
                        len(self.next_event_is_step)) and
                len(self.step_idx) > 10):
            self.ok = False

            # there may be other issues,and bugs,add more tests here if you like

    def save(self, par):
        '''saves track next to its original csv-file'''

        if self.ok:
            file_path = '/'.join((par['parent_dir'], self.experiment,
                                  par['tracked_on'],
                                  par['group_name'],
                                  self.condition, self.trial,
                                  str(self.track_number) + par['path_suffix']))
            some_file = open(file_path, 'wb')
            cPickle.dump(self, some_file, protocol=2)
            # pickle.dump(self, some_file, protocol=2)
            some_file.close()

    def animate_track_and_vars(self, par, speed=1.0, zoom_dx=3,
                               save_movie=False,
                               movie_name='some_movie_name'):
        '''animates track'''
        degree_sign = 'degree'
        font_size = 12
        old_dpi = plt.rcParams['savefig.dpi']
        plt.rcParams['savefig.dpi'] = par['animation_dpi']
        plotTime = 10.0
        # figure settings
        W = 4 * par['fig_width']
        H = 2 * par['fig_width']
        if (int(par['animation_dpi'] * W) % 2 == 1):
            W = (par['animation_dpi'] * W + 1) / par['animation_dpi']

        if (int(par['animation_dpi'] * H) % 2 == 1):
            H = (par['animation_dpi'] * H + 1) / par['animation_dpi']

        fig = plt.figure(figsize=(W, H))

        gs1 = gridspec.GridSpec(5, 2,
                                width_ratios=[2, 1]
                                )
        # gs1.update(left=0.02, right=0.98,
        #           hspace=0.1, wspace=0.1, bottom=0.02, top=0.98)
        ax1 = plt.subplot(gs1[:, :1])
        ax2 = plt.subplot(gs1[0, 1:2])
        ax3 = plt.subplot(gs1[1, 1:2])
        ax4 = plt.subplot(gs1[2, 1:2])
        ax5 = plt.subplot(gs1[3, 1:2])
        ax6 = plt.subplot(gs1[4, 1:2])

        # init variable plots
        head_vector_ang_line, = ax2.plot([], [], lw=1, color='sienna')
        back_vector_ang_line, = ax3.plot([], [], lw=1, color='royalblue')
        heading_angle_line, = ax4.plot([], [], lw=1, color='sienna')
        bearing_angle_line, = ax5.plot([], [], lw=1, color='royalblue')
        bending_angle_line, = ax6.plot([], [], lw=1, color='sienna')

        head_vector_ang_speed_flipped = -1.0 * np.rad2deg(
            self.head_vector_angular_speed)
        plot_vars = [
            head_vector_ang_speed_flipped,
            np.rad2deg(self.back_vector_angular_speed),
            np.rad2deg(self.heading_angle),
            np.rad2deg(self.bearing_angle),
            np.rad2deg(self.bending_angle)
        ]

        # Tail vector is back vector
        variable_names = [
            'Head vector\nangular speed (' + degree_sign + '/s)',
            'Tail vector\nangular speed (' + degree_sign + '/s)',
            'Heading angle (' + degree_sign + ')',
            'Bearing angle (' + degree_sign + ')',
            'Bending angle (' + degree_sign + ')'
        ]
        for i, ax in enumerate(fig.axes):
            if i > 0:
                ax.axhline(0, color='gray', lw=1.0, zorder=-10)
                # plot threshold bending_angle_diff
                if i == 1:
                    ax.axhline(np.rad2deg(
                        par['threshold_head_vector_angular_speed']),
                        color='r', ls='--', alpha=0.6, lw=0.8, zorder=-10)
                    ax.axhline(-np.rad2deg(
                        par['threshold_head_vector_angular_speed']),
                               color='g', ls='--', alpha=0.6, lw=0.8, zorder=-10)

                # plot threshold max of back vector angular speed
                if i == 2:
                    ax.axhline(np.rad2deg(
                        par['threshold_back_vector_angular_speed']),
                        lw=0.8,
                        alpha=0.6,
                        color='b',
                        ls='--')

                [ax.spines[str_tmp].set_color('none')
                 for str_tmp in ['top', 'right']]
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                # ax.xaxis.set_major_locator(MaxNLocator(5))
                # ax.yaxis.set_major_locator(MaxNLocator(4))
                plt.setp(ax, xlabel='Time (s) ',
                         xlim=(np.nanmin(self.time),
                               np.nanmin(self.time) + plotTime))
                if (np.min(plot_vars[i - 1]) > 0):
                    ymin = np.nanmin(plot_vars[i - 1])
                    ymax = np.nanmax(plot_vars[i - 1])
                else:
                    ymin = -1.0 * np.maximum(abs(np.nanmin(plot_vars[i - 1])),
                                             np.nanmax(plot_vars[i - 1]))
                    ymax = np.nanmax(plot_vars[i - 1])
                if (i == 2):
                    ymin = 0
                    ymax = np.nanmax(plot_vars[i - 1])

                plt.setp(ax, ylabel=variable_names[i - 1],
                         ylim=(1.1 * ymin, 1.1 * ymax))
                # for tl in ax.get_xticklabels() + ax.get_yticklabels():
                #    tl.set_visible(False)

        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 ylim=(-par['radius_dish'] - 5,
                       par['radius_dish'] + 5), xticks=[], yticks=[])

        # plot edge of petri dish
        # patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
        #              fc='k', lw=0, alpha=0.1)
        # ax1.add_artist(patch)

        # plot odor
        # ax1.plot(par['odor_A'][0], par['odor_A'][1], ls='none',
        #          marker='o', ms=30,
        #          mec='lightgray', mfc='None', mew=10., alpha=0.1)

        # time text
        time_text = ax1.text(x=0.03, y=0.03, s='',
                             size=font_size, horizontalalignment='left',
                             verticalalignment='bottom',
                             alpha=1, transform=ax1.transAxes)

        # valid text
        valid_text = ax1.text(x=0.03, y=0.93, s='', size=font_size,
                              horizontalalignment='left',
                              verticalalignment='bottom',
                              alpha=1, color='r', transform=ax1.transAxes)

        # back_vector_orthogonal
        back_vector_orthogonal = np.array(mf.rotate_vector_clockwise(
            np.pi / 2.,
            [self.back_vector[:, i] for i in [0, 1]])).T
        print(back_vector_orthogonal.shape)

        # init subplot
        tail_line, = ax1.plot([], [], color='royalblue', lw=2)
        head_line, = ax1.plot([], [], color='sienna', lw=2)

        head_line_left, = ax1.plot([], [], 'g-', alpha=0.5, lw=10)
        head_line_right, = ax1.plot([], [], 'r-', alpha=0.5, lw=10)

        current_spine, = ax1.plot([], [], 'k-', lw=1)

        contour_line = Polygon(np.nan * np.zeros((2, 2)), lw=1,
                               fc='lightgray', ec='k')
        ax1.add_artist(contour_line)

        # HC left and right
        HC_left = np.zeros((len(self.time), 1)) * np.nan
        HC_right = np.zeros((len(self.time), 1)) * np.nan

        if hasattr(self, 'HC_start_idx'):
            for start_idx, end_idx, HC_dir in zip(self.HC_start_idx,
                                                  self.HC_end_idx,
                                                  self.HC_direction):
                if HC_dir == -1:
                    HC_left[start_idx: end_idx, 0] = 1.
                if HC_dir == 1:
                    HC_right[start_idx: end_idx, 0] = 1.

        # initialization function: plot the background of each frame
        def init():
            tail_line.set_data([], [])
            head_line.set_data([], [])
            head_line_left.set_data([], [])
            head_line_right.set_data([], [])
            current_spine.set_data([], [])
            contour_line.set_xy(np.nan * np.zeros((2, 2)))

            # colors = {-1: 'g', 1: 'r'}
            # for idx, ax in enumerate(fig.axes):
            #     if idx > 0:
            #         for start_idx, end_idx, direction in zip(
            #                 self.HC_start_idx,
            #                 self.HC_end_idx,
            #                 self.HC_direction):
            #             ax.axvspan(self.time[start_idx],
            #                        self.time[end_idx],
            #                        facecolor=colors[direction],
            #                         alpha=0.2, lw=0)

            # init variable data
            head_vector_ang_line.set_data([], [])
            back_vector_ang_line.set_data([], [])
            heading_angle_line.set_data([], [])
            bearing_angle_line.set_data([], [])
            bending_angle_line.set_data([], [])

            return (tail_line, head_line, head_line_left, head_line_right,
                    current_spine, contour_line, bending_angle_line,
                    head_vector_ang_line, back_vector_ang_line,
                    bearing_angle_line,
                    heading_angle_line)

        # animation function
        def animate(i):
            limmin = max(self.time[i] - 0.75 * plotTime, np.nanmin(self.time))
            limmax = limmin + plotTime
            idxmin = max(0, i - (0.75 * plotTime * par['fps']))
            for idx, ax in enumerate(fig.axes):
                if idx > 0:
                    # ax.xaxis.set_major_locator(MaxNLocator(6))
                    # ax.yaxis.set_major_locator(MaxNLocator(4))
                    # plot HCs
                    plt.setp(ax, xlabel='Time (s) ',
                             xlim=(limmin, limmax))
                    colors = {-1: 'g', 1: 'r'}

                    for start_idx, end_idx, direction in zip(
                            self.HC_start_idx,
                            self.HC_end_idx,
                            self.HC_direction):
                        if end_idx > i and start_idx < i:
                            ax.axvspan(self.time[i - 1],
                                       self.time[i],
                                       facecolor=colors[direction],
                                       alpha=0.2, lw=0)

            tail_line.set_data(self.spine[1][:i, 0], self.spine[1][:i, 1])
            head_line.set_data(self.spine[11][:i, 0], self.spine[11][:i, 1])
            head_line_left.set_data((HC_left * self.spine[11])[:i, 0],
                                    self.spine[11][:i, 1])
            head_line_right.set_data((HC_right * self.spine[11])[:i, 0],
                                     self.spine[11][:i, 1])
            current_spine.set_data(
                [self.spine[idx][i, 0] for idx in range(12)],
                [self.spine[idx][i, 1] for idx in range(12)]
            )
            contour_line.set_xy(np.array([list(self.contour[idx][i, :])
                                          for idx in list(range(22)) + [0]]))
            # variable plots
            head_vector_ang_line.set_data(
                self.time[idxmin:i],
                plot_vars[0][idxmin:i])
            back_vector_ang_line.set_data(
                self.time[idxmin:i],
                plot_vars[1][idxmin:i])
            heading_angle_line.set_data(self.time[idxmin:i],
                                        plot_vars[2][idxmin:i])
            bearing_angle_line.set_data(self.time[idxmin:i],
                                        plot_vars[3][idxmin:i])
            bending_angle_line.set_data(self.time[idxmin:i],
                                        plot_vars[4][idxmin:i])

            # time text
            time_text.set_text(str(np.round(self.time[i], 1)) + ' seconds')

            # valid text
            if np.isnan(self.valid_frame[i]):
                valid_text.set_text('Invalid frame')
            else:
                valid_text.set_text('')

            # zoom in
            if True:
                plt.setp(ax1,
                         xlim=(self.spine[4][i, 0] - zoom_dx,
                               self.spine[4][i, 0] + zoom_dx),
                         ylim=(self.spine[4][i, 1] - zoom_dx,
                               self.spine[4][i, 1] + zoom_dx))

            # if step
            # if i in self.step_idx:
            #     ax1.plot(
            #         [self.spine[0][i, 0] - 0.5 * back_vector_orthogonal[i, 0],
            #         self.spine[0][i, 0] + 0.5 * back_vector_orthogonal[i, 0]],
            #         [self.spine[0][i, 1] - 0.5 * back_vector_orthogonal[i, 1],
            #         self.spine[0][i, 1] + 0.5 * back_vector_orthogonal[i, 1]],
            #         'k-', alpha=0.2, lw=4)

            return (tail_line, head_line, head_line_left, head_line_right,
                    current_spine, contour_line, bending_angle_line,
                    head_vector_ang_line, back_vector_ang_line,
                    bearing_angle_line,
                    heading_angle_line)

        global ani
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.time),
            interval=int(1000. / speed) * float(par['dt']),
            repeat=False)

        # save or show movie
        if save_movie:
            print('Saving movie...')

            # bitrate = 100 - 600 works fine
            mywriter = animation.AVConvFileWriter(
                np.round(.5 / float(par['dt'])),
                bitrate=600,
                codec='libx264')

            ani.save(par['movie_dir'] +
                     '/' + movie_name + '.mp4',
                     writer=mywriter)

            print('...done')

        plt.rcParams['savefig.dpi'] = old_dpi

    def animate_track(self, par, speed=1.0, zoom_dx=3, save_movie=False,
                      movie_name='some_movie_name'):
        '''animates track'''

        old_dpi = plt.rcParams['savefig.dpi']
        plt.rcParams['savefig.dpi'] = par['animation_dpi']
        # figure settings
        fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=0.02, right=0.98,
                   hspace=0.1, wspace=0.1, bottom=0.02, top=0.98)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 ylim=(-par['radius_dish'] - 5,
                       par['radius_dish'] + 5), xticks=[], yticks=[])

        # plot edge of petri dish
        patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
                      fc='k', lw=0, alpha=0.1)
        ax1.add_artist(patch)

        # plot odor
        ax1.plot(par['odor_A'][0], par['odor_A'][1], ls='none',
                 marker='o', ms=30,
                 mec='lightgray', mfc='None', mew=10., alpha=0.1)

        # time text
        time_text = ax1.text(x=0.03, y=0.03, s='',
                             size=font_size, horizontalalignment='left',
                             verticalalignment='bottom',
                             alpha=1, transform=ax1.transAxes)

        # valid text
        valid_text = ax1.text(x=0.03, y=0.93, s='', size=font_size,
                              horizontalalignment='left',
                              verticalalignment='bottom',
                              alpha=1, color='r', transform=ax1.transAxes)

        # back_vector_orthogonal
        back_vector_orthogonal = np.array(rotate_vector_clockwise(
            np.pi / 2.,
            [self.back_vector[:, i] for i in [0, 1]])).T
        print(back_vector_orthogonal.shape)

        # init subplot
        midpoint_line, = ax1.plot([], [], 'k-', alpha=0.5, lw=2)
        head_line, = ax1.plot([], [], 'm-', alpha=0.5, lw=2)

        head_line_left, = ax1.plot([], [], 'g-', alpha=0.5, lw=10)
        head_line_right, = ax1.plot([], [], 'r-', alpha=0.5, lw=10)

        current_spine, = ax1.plot([], [], 'k-', lw=1)

        contour_line = Polygon(np.nan * np.zeros((2, 2)), lw=1,
                               fc='lightgray', ec='k')
        ax1.add_artist(contour_line)

        # HC left and right
        HC_left = np.zeros((len(self.time), 1)) * np.nan
        HC_right = np.zeros((len(self.time), 1)) * np.nan
        if hasattr(self, 'HC_start_idx'):
            for start_idx, end_idx, HC_dir in zip(self.HC_start_idx,
                                                  self.HC_end_idx,
                                                  self.HC_direction):
                if HC_dir == -1:
                    HC_left[start_idx: end_idx, 0] = 1.
                if HC_dir == 1:
                    HC_right[start_idx: end_idx, 0] = 1.

        # initialization function: plot the background of each frame
        def init():
            midpoint_line.set_data([], [])
            head_line.set_data([], [])
            head_line_left.set_data([], [])
            head_line_right.set_data([], [])
            current_spine.set_data([], [])
            contour_line.set_xy(np.nan * np.zeros((2, 2)))
            return (midpoint_line, head_line, head_line_left, head_line_right,
                    current_spine, contour_line,)

        # animation function
        def animate(i):

            midpoint_line.set_data(self.spine[4][:i, 0], self.spine[4][:i, 1])
            head_line.set_data(self.spine[11][:i, 0], self.spine[11][:i, 1])
            head_line_left.set_data((HC_left * self.spine[11])[:i, 0],
                                    self.spine[11][:i, 1])
            head_line_right.set_data((HC_right * self.spine[11])[:i, 0],
                                     self.spine[11][:i, 1])
            current_spine.set_data(
                [self.spine[idx][i, 0] for idx in range(12)],
                [self.spine[idx][i, 1] for idx in range(12)]
            )
            contour_line.set_xy(np.array([list(self.contour[idx][i, :])
                                          for idx in list(range(22)) + [0]]))
            # time text
            time_text.set_text(str(np.round(self.time[i], 1)) + ' seconds')

            # valid text
            if np.isnan(self.valid_frame[i]):
                valid_text.set_text('Invalid frame')
            else:
                valid_text.set_text('')

            # zoom in
            if True:
                plt.setp(ax1,
                         xlim=(self.spine[4][i, 0] - zoom_dx,
                               self.spine[4][i, 0] + zoom_dx),
                         ylim=(self.spine[4][i, 1] - zoom_dx,
                               self.spine[4][i, 1] + zoom_dx))

            # if step
            if i in self.step_idx:
                ax1.plot(
                    [self.spine[0][i, 0] - 0.5 * back_vector_orthogonal[i, 0],
                     self.spine[0][i, 0] + 0.5 * back_vector_orthogonal[i, 0]],
                    [self.spine[0][i, 1] - 0.5 * back_vector_orthogonal[i, 1],
                     self.spine[0][i, 1] + 0.5 * back_vector_orthogonal[i, 1]],
                    'k-', alpha=0.2, lw=4)

            return (midpoint_line, head_line, current_spine, contour_line,)

        global ani
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.time),
            interval=int(1000. / speed) * float(par['dt']),
            repeat=False)

        # save or show movie
        if save_movie:
            print('Saving movie...')

            # bitrate = 100 - 600 works fine
            mywriter = animation.AVConvFileWriter(
                np.round(1. / float(par['dt'])),
                bitrate=600,
                codec='libx264')

            ani.save(par['movie_dir'] +
                     '/' + movie_name + '.mp4',
                     writer=mywriter)

            print('...done')

        plt.rcParams['savefig.dpi'] = old_dpi

    def figure_track_on_dish_whole(
            self, par, keys=[],
            figure_name='single_track_on_dish'):

        # figure settings
        fig = plt.figure(figsize=(1.5 * par['fig_width'],
                                  1.5 * par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=.02, right=.98, hspace=.1,
                   wspace=.1, bottom=.02, top=.98)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[
             str_tmp].set_color('none') for str_tmp in ['top', 'right',
                                                        'left', 'bottom']]
        plt.setp(ax1, xlim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 ylim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 xticks=[], yticks=[])

        # plot edge of petri dish
        patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
                      fc='k', lw=0, alpha=0.1)
        ax1.add_artist(patch)

        # plot odor
        ax1.plot(par['odor_A'][0], par['odor_A'][1], ls='none',
                 color='k', marker='o',
                 ms=20, mec='lightgray', mfc='none', mew=10., alpha=0.9)

        # plot spine point 5 (= spine point 6 in the manuskript)
        ax1.plot(self.spine[5][:, 0], self.spine[5][:, 1], lw=1, ls='-',
                 color='royalblue', alpha=0.7, label='Midpoint')

        # plot head
        # ax1.plot(self.spine[11][:, 0], self.spine[11][:, 1], lw=1, ls='-',
        #         color='sienna', alpha=1, label='Head')

        # plot start and end contour line
        if 'start' in keys:
            ax1.plot([self.spine[idx][0, 0] for idx in range(12)],
                     [self.spine[idx][0, 1] for idx in range(12)],
                     'k-', alpha=0.3, lw=1)
            contour_line = Polygon(
                np.array([list(self.contour[idx][0, :])
                          for idx in list(range(22)) + [0]]),
                alpha=0.1, ec='k', fc='k')
            ax1.add_artist(contour_line)
        if 'end' in keys:
            ax1.plot([self.spine[idx][-1, 0] for idx in range(12)],
                     [self.spine[idx][-1, 1] for idx in range(12)],
                     'k-', alpha=0.3, lw=1)
            contour_line = Polygon(np.array([list(self.contour[idx][-1, :])
                                             for idx in list(range(22)) + [0]]),
                                   alpha=0.1, ec='k', fc='k')
            ax1.add_artist(contour_line)

        # plot steps
        if 'steps' in keys:
            back_vector_orthogonal = np.array(rotate_vector_clockwise(
                np.pi / 2.,
                [self.back_vector[:, i] for i in [0, 1]])).T
            #  for idx in self.step_idx:
            #      ax1.plot(
            #          [self.spine[0][idx, 0] - back_vector_orthogonal[idx, 0],
            #           self.spine[0][idx, 0] + back_vector_orthogonal[idx, 0]],
            #          [self.spine[0][idx, 1] - back_vector_orthogonal[idx, 1],
            #           self.spine[0][idx, 1] + back_vector_orthogonal[idx, 1]],
            #          'k-', alpha=0.3, lw=2)

            # annotate INS_turn
            if 'INS_turn' in keys:
                for step_count in range(len(self.step_idx) - 1):
                    str_xy = (
                            (self.spine[0][self.step_idx[step_count], :] +
                             self.spine[0][self.step_idx[step_count + 1], :]) / 2. +
                            0.5 *
                            back_vector_orthogonal[self.step_idx[step_count], :])
                    ax1.annotate(str(np.round(
                        180. / np.pi * self.INS_turn[step_count], 1)) +
                                 degree_sign,
                                 xy=str_xy, size=font_size, horizontalalignment='center',
                                 verticalalignment='center', color='k')

                    # fake step for legend
                    # ax1.plot(
                    # [100, 100],[100, 100],'k-', alpha=.3, lw=4, label='Step')

        # plot INS_interval
        if 'INS_distance' in keys:
            ax1.annotate(
                '', xy=(self.spine[0][self.step_idx[12], 0],
                        self.spine[0][self.step_idx[12], 1]),
                xycoords='data',
                xytext=(self.spine[0][self.step_idx[13], 0],
                        self.spine[0][self.step_idx[13], 1]),
                textcoords='data',
                arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                alpha=1, color='k', lw=2))
            ax1.annotate(
                'Inter-step-distance', xy=(
                    self.spine[0][self.step_idx[12], 0] + 0,
                    self.spine[0][self.step_idx[12], 1] + 1),
                xycoords='data', size=font_size,
                horizontalalignment='center',
                verticalalignment='center')

        # plot HCs
        if 'HCs' in keys:
            HC_left = np.zeros(len(self.time)) * np.nan
            HC_right = np.zeros(len(self.time)) * np.nan
            for start_idx, end_idx, HC_dir, in zip(self.HC_start_idx,
                                                   self.HC_end_idx,
                                                   self.HC_direction):
                if HC_dir == -1:
                    HC_left[start_idx: end_idx] = 1.
                if HC_dir == 1:
                    HC_right[start_idx: end_idx] = 1.

            ax1.plot(HC_left * self.spine[11][:, 0], self.spine[11][:, 1],
                     'g-', alpha=0.3, lw=10, label='Head cast left')
            ax1.plot(HC_right * self.spine[11][:, 0], self.spine[11][:, 1],
                     'r-', alpha=0.3, lw=10, label='Head cast right')

            # annotate HC angles
            # colors = {-1: 'g', 1: 'r'}
            # if any true HCs
            # if len(self.HC_start_idx) > 0:
            #     for HC_idx,start_idx in zip(np.arange(self.HC_start_idx.size),
            #                                  self.HC_start_idx):
            #         str_xy = (self.spine[11][start_idx, :] +
            #                   0.5 * back_vector_orthogonal[start_idx, :])
            #         ax1.annotate(
            #             str(np.round(180./np.pi * self.HC_angle[HC_idx], 1)) +
            #             degree_sign,
            #             xy=str_xy,size=font_size,horizontalalignment='center',
            #             verticalalignment='center',
            #             color=colors[self.HC_direction[HC_idx]])

        # zoom in
        # delta = 1.0
        # dx = np.nanmax(self.spine[5][:, 0]) - np.nanmin(self.spine[5][:, 0])
        # dy = np.nanmax(self.spine[5][:, 1]) - np.nanmin(self.spine[5][:, 1])
        # dxy = np.max([dx, dy])
        # plt.setp(ax1,
        #          xlim=([-100, 100]),
        #          ylim=([-100, 100])
        #          )
        # plt.setp(ax1,
        #          xlim=(np.nanmin(self.spine[5][:, 0]) -
        #                delta, np.nanmin(self.spine[5][:, 0]) + dxy + delta),
        #          ylim=(np.nanmin(self.spine[5][:, 1]) -
        #                delta, np.nanmin(self.spine[5][:, 1]) + dxy + delta)
        #          )

        # legend
        leg = ax1.legend(loc=[0.0, 0.0], ncol=0, handlelength=2)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)

        # save plot
        if par['save_figure']:
            print("Track on dish: " + str(par['figure_dir']) + '/' + figure_name)
            plt.savefig(par['figure_dir'] +
                        '/' + figure_name)
            plt.close()

    def figure_track_on_dish(self, par, keys=[],
                             figure_name='single_track_on_dish'):

        # figure settings
        fig = plt.figure(figsize=(1.5 * par['fig_width'],
                                  1.5 * par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=.02, right=.98, hspace=.1,
                   wspace=.1, bottom=.02, top=.98)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[
             str_tmp].set_color('none') for str_tmp in ['top', 'right',
                                                        'left', 'bottom']]
        plt.setp(ax1, xlim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 ylim=(-par['radius_dish'] - 5, par['radius_dish'] + 5),
                 xticks=[], yticks=[])

        # plot edge of petri dish
        patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
                      fc='k', lw=0, alpha=0.1)
        ax1.add_artist(patch)

        # plot odor
        ax1.plot(par['odor_A'][0], par['odor_A'][1], ls='none',
                 color='k', marker='o',
                 ms=20, mec='lightgray', mfc='none', mew=10., alpha=0.9)

        # plot spine point 5 (= spine point 6 in the manuskript)
        ax1.plot(self.spine[1][:, 0], self.spine[1][:, 1], lw=1, ls='-',
                 color='royalblue', alpha=0.7, label='Tail')

        # plot head
        ax1.plot(self.spine[11][:, 0], self.spine[11][:, 1], lw=1, ls='-',
                 color='sienna', alpha=1, label='Head')

        # plot start and end contour line
        if 'start' in keys:
            ax1.plot([self.spine[idx][0, 0] for idx in range(12)],
                     [self.spine[idx][0, 1] for idx in range(12)],
                     'k-', alpha=0.3, lw=1)
            contour_line = Polygon(
                np.array([list(self.contour[idx][0, :])
                          for idx in list(range(22)) + [0]]),
                alpha=0.1, ec='k', fc='k')
            ax1.add_artist(contour_line)
        if 'end' in keys:
            ax1.plot([self.spine[idx][-1, 0] for idx in range(12)],
                     [self.spine[idx][-1, 1] for idx in range(12)],
                     'k-', alpha=0.3, lw=1)
            contour_line = Polygon(np.array([list(self.contour[idx][-1, :])
                                             for idx in list(range(22)) + [0]]),
                                   alpha=0.1, ec='k', fc='k')
            ax1.add_artist(contour_line)

        # plot steps
        if 'steps' in keys:
            back_vector_orthogonal = np.array(mf.rotate_vector_clockwise(
                np.pi / 2.,
                [self.back_vector[:, i] for i in [0, 1]])).T
            #  for idx in self.step_idx:
            #      ax1.plot(
            #          [self.spine[0][idx, 0] - back_vector_orthogonal[idx, 0],
            #           self.spine[0][idx, 0] + back_vector_orthogonal[idx, 0]],
            #          [self.spine[0][idx, 1] - back_vector_orthogonal[idx, 1],
            #           self.spine[0][idx, 1] + back_vector_orthogonal[idx, 1]],
            #          'k-', alpha=0.3, lw=2)

            # annotate INS_turn
            if 'INS_turn' in keys:
                for step_count in range(len(self.step_idx) - 1):
                    str_xy = (
                            (self.spine[0][self.step_idx[step_count], :] +
                             self.spine[0][self.step_idx[step_count + 1], :]) / 2. +
                            0.5 *
                            back_vector_orthogonal[self.step_idx[step_count], :])
                    ax1.annotate(str(np.round(
                        180. / np.pi * self.INS_turn[step_count], 1)) +
                                 degree_sign,
                                 xy=str_xy, size=font_size, horizontalalignment='center',
                                 verticalalignment='center', color='k')

                    # fake step for legend
                    # ax1.plot(
                    # [100, 100],[100,100], 'k-', alpha=.3, lw=4, label='Step')

        # plot INS_interval
        if 'INS_distance' in keys:
            ax1.annotate(
                '', xy=(self.spine[0][self.step_idx[12], 0],
                        self.spine[0][self.step_idx[12], 1]),
                xycoords='data',
                xytext=(self.spine[0][self.step_idx[13], 0],
                        self.spine[0][self.step_idx[13], 1]),
                textcoords='data',
                arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                alpha=1, color='k', lw=2))
            ax1.annotate(
                'Inter-step-distance', xy=(
                    self.spine[0][self.step_idx[12], 0] + 0,
                    self.spine[0][self.step_idx[12], 1] + 1),
                xycoords='data', size=font_size,
                horizontalalignment='center',
                verticalalignment='center')

        # plot HCs
        if 'HCs' in keys:
            HC_left = np.zeros(len(self.time)) * np.nan
            HC_right = np.zeros(len(self.time)) * np.nan
            for start_idx, end_idx, HC_dir, in zip(self.HC_start_idx,
                                                   self.HC_end_idx,
                                                   self.HC_direction):
                if HC_dir == -1:
                    HC_left[start_idx: end_idx] = 1.
                if HC_dir == 1:
                    HC_right[start_idx: end_idx] = 1.

            ax1.plot(HC_left * self.spine[11][:, 0], self.spine[11][:, 1],
                     'g-', alpha=0.3, lw=10, label='Head cast left')
            ax1.plot(HC_right * self.spine[11][:, 0], self.spine[11][:, 1],
                     'r-', alpha=0.3, lw=10, label='Head cast right')

            # annotate HC angles
            # colors = {-1: 'g', 1: 'r'}
            # if any true HCs
            # if len(self.HC_start_idx) > 0:
            #     for HC_idx,start_idx in zip(np.arange(self.HC_start_idx.size),
            #                                  self.HC_start_idx):
            #         str_xy = (self.spine[11][start_idx, :] +
            #                   0.5 * back_vector_orthogonal[start_idx, :])
            #         ax1.annotate(
            #             str(np.round(180./np.pi * self.HC_angle[HC_idx], 1)) +
            #             degree_sign,
            #             xy=str_xy,size=font_size,horizontalalignment='center',
            #             verticalalignment='center',
            #             color=colors[self.HC_direction[HC_idx]])

        # zoom in
        delta = 4
        dx = np.nanmax(self.spine[5][:, 0]) - np.nanmin(self.spine[5][:, 0])
        dy = np.nanmax(self.spine[5][:, 1]) - np.nanmin(self.spine[5][:, 1])
        dxy = np.max([dx, dy])
        # plt.setp(ax1,
        #          xlim=([-100, 100]),
        #          ylim=([-100, 100])
        #          )
        plt.setp(ax1,
                 xlim=(np.nanmin(self.spine[5][:, 0]) -
                       delta, np.nanmin(self.spine[5][:, 0]) + dxy + delta),
                 ylim=(np.nanmin(self.spine[5][:, 1]) -
                       delta, np.nanmin(self.spine[5][:, 1]) + dxy + delta)
                 )

        # legend
        leg = ax1.legend(loc=[0.1, 0.8], ncol=1, handlelength=2)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)

        # save plot
        if par['save_figure']:
            print("Track on dish: " + str(par['figure_dir']) + '/' + figure_name)
            plt.savefig(par['figure_dir'] +
                        '/' + figure_name)
            plt.close()

    def figure_single_contour(self, par, i,
                              keys, figure_name='single_contour'):

        # figure settings
        fig = plt.figure(figsize=(0.4 * par['fig_width'],
                                  0.4 * par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=0.02, right=0.98, hspace=0.1,
                   wspace=0.1, bottom=0.02, top=0.98)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[str_tmp].set_color('none') for str_tmp in [
            'top', 'right',
            'left', 'bottom']]
        plt.setp(ax1, xticks=[], yticks=[])

        # plot spine
        ax1.plot([self.spine[idx][i, 0] for idx in range(12)],
                 [self.spine[idx][i, 1] for idx in range(12)],
                 color='gray', alpha=1, lw=1, marker='.', ms=7,
                 mec='None', mfc='gray')

        # plot contour
        contour_line = Polygon(np.array([list(self.contour[idx][i, :])
                                         for idx in list(range(22)) + [0]]),
                               alpha=0.1, ec='none', fc='k')
        ax1.add_artist(contour_line)

        # plot contour line
        contour_line = ax1.plot(
            [self.contour[idx][i, 0] for idx in list(range(22)) + [0]],
            [self.contour[idx][i, 1] for idx in list(range(22)) + [0]],
            color='gray')

        # add numbers
        if False:
            for idx in range(12):
                str_xy = self.spine[idx][i, :] - np.array([.3, .3])
                ax1.annotate(str(idx), xy=str_xy, size=font_size,
                             horizontalalignment='center',
                             verticalalignment='center', color='k')

        if 'back' in keys:
            ax1.annotate(
                '', xy=(self.spine[5][i, 0], self.spine[5][i, 1]),
                xycoords='data',
                xytext=(self.spine[1][i, 0], self.spine[1][i, 1]),
                textcoords='data',
                arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                                alpha=1, color='k', lw=1.5))

        if 'front' in keys:
            ax1.annotate(
                '', xy=(self.spine[10][i, 0], self.spine[10][i, 1]),
                xycoords='data',
                xytext=(self.spine[5][i, 0], self.spine[5][i, 1]),
                textcoords='data',
                arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                                alpha=1, color='k', lw=1.5))
        if 'head' in keys:
            ax1.annotate(
                '', xy=(self.spine[10][i, 0], self.spine[10][i, 1]),
                xycoords='data',
                xytext=(self.spine[8][i, 0], self.spine[8][i, 1]),
                textcoords='data',
                arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                                alpha=1, color='k', lw=1.5))

        # zoom in
        dx = np.max(self.spine[5][i, 0]) - np.min(self.spine[5][i, 0])
        dy = np.max(self.spine[5][i, 1]) - np.min(self.spine[5][i, 1])
        dxy = np.max([dx, dy])
        plt.setp(ax1,
                 xlim=(np.min(self.spine[5][i, 0]) - 3,
                       np.min(self.spine[5][i, 0]) + dxy + 3),
                 ylim=(np.min(self.spine[5][i, 1]) - 3,
                       np.min(self.spine[5][i, 1]) + dxy + 3)
                 )

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + figure_name)
            plt.close()

    def figure_time_series_variables_sep(self, par, keys=['all_variables'],
                                         figure_name='time_series_variables'):

        font_size = 9
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size

        if 'all_variables' in keys:
            # add or (un)comment variables and respective variable names
            variables = [
                # self.tail_speed_forward,
                np.rad2deg(self.bearing_angle),
                # self.INS_interval[1:]/self.INS_interval[:-1],
                np.rad2deg(self.bending_angle),
                np.rad2deg(self.heading_angle),
                np.rad2deg(self.head_vector_angular_speed),
                np.rad2deg(self.back_vector_angular_speed)
            ]

            variable_names = [
                # 'Tail speed forward (mm/s)',
                # 'current ISI/previous ISI',
                'Bearing angle (' + degree_sign + ')',
                'Bending angle (' + degree_sign + ')',
                'Heading angle (' + degree_sign + ')',
                'Head vector angular speed (' + degree_sign + '/s)',
                'Tail vector angular speed (' + degree_sign + '/s)'
            ]
            variable_colors = [
                'royalblue',
                'sienna',
                'sienna',
                'sienna',
                'royalblue']

        if 'tail_speed' in keys:
            variables = [self.tail_speed_forward]
            variable_names = ['Tail speed forward (mm/s)']

        if 'HC_variables' in keys:
            variables = [
                np.rad2deg(self.head_vector_angular_speed),
                np.rad2deg(self.back_vector_angular_speed)
            ]

            variable_names = [
                'Head vector angular speed (' + degree_sign + '/s)',
                'Tail vector angular speed (' + degree_sign + '/s)'
            ]

        # number of subplots
        n_subplots = len(variables)

        fig1 = plt.figure(0)
        fig1.set_size_inches(1.5 * par['fig_width'],
                             0.75 * par['fig_width'])
        fig2 = plt.figure(1)
        fig2.set_size_inches(1.5 * par['fig_width'],
                             0.75 * par['fig_width'])
        fig3 = plt.figure(2)
        fig3.set_size_inches(1.5 * par['fig_width'],
                             0.75 * par['fig_width'])
        fig4 = plt.figure(3)
        fig4.set_size_inches(1.5 * par['fig_width'],
                             0.75 * par['fig_width'])
        fig5 = plt.figure(4)
        fig5.set_size_inches(1.5 * par['fig_width'],
                             0.75 * par['fig_width'])
        figs = [fig1, fig2, fig3, fig4, fig5]

        # for all subplots
        for subplot_idx in range(n_subplots):
            pltvar = variables[subplot_idx] * self.valid_frame
            shiftTime = self.time[:len(pltvar)] - min(self.time[:len(pltvar)])
            minTime = np.nanmin(self.time[:len(pltvar)])
            ax = figs[subplot_idx].add_subplot(111)
            figs[subplot_idx].subplots_adjust(top=0.95, bottom=0.15)
            [ax.spines[str_tmp].set_color('none')
             for str_tmp in ['top', 'right']]
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            plt.setp(ax, xlabel='Time (s)', xlim=(shiftTime[0], shiftTime[-1]))

            # ylabel
            ax.annotate(variable_names[subplot_idx], xy=(-0.12, .5),
                        xycoords='axes fraction', size=font_size,
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation=90)

            # plot foward step
            if 'steps' in keys:
                for idx in self.step_idx:
                    ax.axvline(shiftTime, lw=2, color='k', alpha=0.2)

            # plot HCs
            if 'HCs' in keys:
                colors = {-1: 'g', 1: 'r'}
                for start_idx, end_idx, direction in zip(self.HC_start_idx,
                                                         self.HC_end_idx,
                                                         self.HC_direction):
                    ax.axvspan(self.time[start_idx] - minTime,
                               self.time[end_idx] - minTime,
                               facecolor=colors[direction], alpha=0.2, lw=0)

            # plot blue circles and tail_speed_threshold
            # if 'Tail speed' in variable_names[subplot_idx] ==\
            #         'Tail speed forward (mm/s)':
            #     ax.plot(self.time[self.step_idx],
            #             (self.tail_speed_forward *
            #              self.valid_frame)[self.step_idx],
            #             mec='b', ls='', mew=1, marker='o', ms=8, mfc='None')
            #     ax.axhline(par['threshold_tail_speed_forward'],
            #                color='blue', ls='--')

            # if 'Midpoint speed' in variable_names[subplot_idx] ==\
            #        'Midpoint speed (mm/s)':
            #     ax.plot(self.time[self.step_idx],
            #             (self.midpoint_speed *
            #              self.valid_frame)[self.step_idx],
            #             mec='b', ls='', mew=1, ms=4, mfc='None',lw=2)
            #     # ax.axhline(par['threshold_tail_speed_forward'],
            #     #            color='blue', ls='--')

            # plot INS_interval
            if 'INS_interval' in keys:

                # nice ylim
                plt.setp(ax, xlabel='Time (s)', ylim=(-0.2, 2.5))

                # inter-step-intervals
                for step_count in range(len(self.step_idx) - 1):
                    ax.annotate(
                        '', xy=(self.time[self.step_idx[step_count]], 2.4),
                        xycoords='data',
                        xytext=(self.time[self.step_idx[step_count + 1]], 2.4),
                        textcoords='data',
                        arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                        alpha=0.4, color='k', lw=1))

                step_count = 6
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count]], 2.4),
                    xycoords='data',
                    xytext=(self.time[self.step_idx[step_count + 1]], 2.4),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                    alpha=1, color='k', lw=1))

                ax.annotate(
                    'Inter-step-interval', xy=((
                                                       self.time[self.step_idx[step_count]] +
                                                       self.time[self.step_idx[step_count + 1]]) / 2, 2.5),
                    xycoords='data', size=font_size,
                    horizontalalignment='center')

                # indicate step motion
                idx_start = self.step_idx[2] - 5
                idx_end = self.step_idx[2] + 5
                ax.fill_between(self.time[idx_start: idx_end],
                                np.zeros(idx_end - idx_start),
                                self.tail_speed_forward[idx_start: idx_end],
                                facecolor='b', alpha=0.2, lw=0,
                                edgecolor='None')

            # plot step_HC_interval
            if 'step_HC_interval' in keys:
                step_count = 7
                HC_count = 3
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count]], 280),
                    xycoords='data',
                    xytext=(self.time[self.HC_start_idx[HC_count]], 280),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0,
                                    shrinkB=0,
                                    alpha=1, color='k', lw=1))
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count + 1]], 280),
                    xycoords='data',
                    xytext=(self.time[self.HC_start_idx[HC_count]], 280),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                    alpha=1, color='k', lw=1))

            # plot variable
            if variable_names[subplot_idx] == 'current ISI/previous ISI':
                ax.plot(self.time[self.step_idx[1:]], variables[subplot_idx],
                        color='b', lw=1)

                # plot one line
                ax.axhline(1, color='gray', ls='--')

            else:

                if (variable_names[subplot_idx] ==
                        'Head vector angular speed (' + degree_sign + '/s)'):
                    pltvar = -1.0 * pltvar
                if (np.min(pltvar) > 0):
                    ymin = np.nanmin(pltvar)
                    ymax = np.nanmax(pltvar)
                else:
                    ymin = -1.0 * np.maximum(abs(np.nanmin(pltvar)),
                                             np.nanmax(pltvar))
                    ymax = np.nanmax(pltvar)
                if (subplot_idx == 4):
                    ymin = 0
                    ymax = np.nanmax(pltvar)

                plt.setp(ax,
                         ylim=(1.1 * ymin, 1.1 * ymax))
                ax.plot(shiftTime,
                        pltvar * self.valid_frame,
                        color=variable_colors[subplot_idx], lw=1.0)

                # plot zero line
                ax.axhline(0, color='gray', lw=1.0, zorder=-10)

            # plot threshold bending_angle_diff
            if variable_names[subplot_idx] == \
                    'Head vector angular speed (' + degree_sign + '/s)':
                ax.axhline(np.rad2deg(
                    par['threshold_head_vector_angular_speed']),
                    color='r', ls='--', lw=1.0, zorder=-10)
                ax.axhline(-np.rad2deg(
                    par['threshold_head_vector_angular_speed']),
                           color='g', ls='--', lw=1.0, zorder=-10)

            # plot threshold max of back vector angular speed
            if variable_names[subplot_idx] == \
                    'Tail vector angular speed (' + degree_sign + '/s)':
                ax.axhline(np.rad2deg(
                    par['threshold_back_vector_angular_speed']),
                    color='b',
                    ls='--')

            # save plot
            if par['save_figure']:
                plt.figure(subplot_idx)
                plt.savefig(par['figure_dir'] +
                            '/' +
                            variable_names[subplot_idx].split(' (')[0])
                plt.close()

        font_size = 7
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size

    def figure_time_series_variables(self, par, keys=['all_variables'],
                                     figure_name='time_series_variables'):

        if 'all_variables' in keys:
            # add or (un)comment variables and respective variable names
            variables = [
                # self.tail_speed_forward,
                self.midpoint_speed,
                # self.INS_interval[1:]/self.INS_interval[:-1],
                np.rad2deg(self.bending_angle),
                np.rad2deg(self.head_vector_angular_speed),
                np.rad2deg(self.back_vector_angular_speed)
            ]

            variable_names = [
                # 'Tail speed forward (mm/s)',
                # 'current ISI/previous ISI',
                'Midpoint speed (mm/s)',
                'Bending angle (' + degree_sign + ')',
                'Head vector angular speed (' + degree_sign + '/s)',
                'Tail vector angular speed (' + degree_sign + '/s)'
            ]

        if 'tail_speed' in keys:
            variables = [self.tail_speed_forward]
            variable_names = ['Tail speed forward (mm/s)']

        if 'HC_variables' in keys:
            variables = [
                np.rad2deg(self.head_vector_angular_speed),
                np.rad2deg(self.back_vector_angular_speed)
            ]

            variable_names = [
                'Head vector angular speed (' + degree_sign + '/s)',
                'Tail vector angular speed (' + degree_sign + '/s)'
            ]

        # number of subplots
        n_subplots = len(variables)

        # figure settings
        fig = plt.figure(figsize=(1.5 * par['fig_width'], 0.8 * n_subplots *
                                  par['fig_width']))
        gs1 = gridspec.GridSpec(n_subplots, 1)
        gs1.update(left=0.15, right=0.95, hspace=0.5, wspace=0.5,
                   bottom=0.15 / n_subplots, top=0.97)
        gs1.get_subplot_params(fig)

        # for all subplots
        for subplot_idx in range(n_subplots):

            ax = plt.subplot(gs1[subplot_idx, 0])
            [ax.spines[str_tmp].set_color('none')
             for str_tmp in ['top', 'right']]
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            plt.setp(ax, xlabel='Time (s)', xlim=(self.time[0], self.time[-1]))

            # ylabel
            ax.annotate(variable_names[subplot_idx], xy=(-0.1, .5),
                        xycoords='axes fraction', size=font_size,
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation=90)

            # plot foward step
            if 'steps' in keys:
                for idx in self.step_idx:
                    ax.axvline(self.time[idx], lw=2, color='k', alpha=0.2)

            # plot HCs
            if 'HCs' in keys:
                colors = {-1: 'g', 1: 'r'}
                for start_idx, end_idx, direction in zip(self.HC_start_idx,
                                                         self.HC_end_idx,
                                                         self.HC_direction):
                    ax.axvspan(self.time[start_idx], self.time[end_idx],
                               facecolor=colors[direction], alpha=0.2, lw=0)

            # plot blue circles and tail_speed_threshold
            # if 'Tail speed' in variable_names[subplot_idx] ==\
            #         'Tail speed forward (mm/s)':
            #     ax.plot(self.time[self.step_idx],
            #             (self.tail_speed_forward *
            #              self.valid_frame)[self.step_idx],
            #             mec='b', ls='', mew=1, marker='o', ms=8, mfc='None')
            #     ax.axhline(par['threshold_tail_speed_forward'],
            #                color='blue', ls='--')

            if 'Midpoint speed' in variable_names[subplot_idx] == \
                    'Midpoint speed (mm/s)':
                ax.plot(self.time[self.step_idx],
                        (self.midpoint_speed *
                         self.valid_frame)[self.step_idx],
                        mec='b', ls='', mew=1, ms=8, mfc='None')
                # ax.axhline(par['threshold_tail_speed_forward'],
                #            color='blue', ls='--')

            # plot INS_interval
            if 'INS_interval' in keys:

                # nice ylim
                plt.setp(ax, xlabel='Time (s)', ylim=(-0.2, 2.5))

                # inter-step-intervals
                for step_count in range(len(self.step_idx) - 1):
                    ax.annotate(
                        '', xy=(self.time[self.step_idx[step_count]], 2.4),
                        xycoords='data',
                        xytext=(self.time[self.step_idx[step_count + 1]], 2.4),
                        textcoords='data',
                        arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                        alpha=0.4, color='k', lw=1))

                step_count = 6
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count]], 2.4),
                    xycoords='data',
                    xytext=(self.time[self.step_idx[step_count + 1]], 2.4),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                    alpha=1, color='k', lw=1))

                ax.annotate(
                    'Inter-step-interval', xy=((
                                                       self.time[self.step_idx[step_count]] +
                                                       self.time[self.step_idx[step_count + 1]]) / 2, 2.5),
                    xycoords='data', size=font_size,
                    horizontalalignment='center')

                # indicate step motion
                idx_start = self.step_idx[2] - 5
                idx_end = self.step_idx[2] + 5
                ax.fill_between(self.time[idx_start: idx_end],
                                np.zeros(idx_end - idx_start),
                                self.tail_speed_forward[idx_start: idx_end],
                                facecolor='b', alpha=0.2, lw=0,
                                edgecolor='None')

            # plot step_HC_interval
            if 'step_HC_interval' in keys:
                step_count = 7
                HC_count = 3
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count]], 280),
                    xycoords='data',
                    xytext=(self.time[self.HC_start_idx[HC_count]], 280),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0,
                                    shrinkB=0,
                                    alpha=1, color='k', lw=1))
                ax.annotate(
                    '', xy=(self.time[self.step_idx[step_count + 1]], 280),
                    xycoords='data',
                    xytext=(self.time[self.HC_start_idx[HC_count]], 280),
                    textcoords='data',
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                                    alpha=1, color='k', lw=1))

            # plot variable
            if variable_names[subplot_idx] == 'current ISI/previous ISI':
                ax.plot(self.time[self.step_idx[1:]], variables[subplot_idx],
                        color='b', lw=1)

                # plot one line
                ax.axhline(1, color='gray', ls='--')

            else:
                ax.plot(self.time[:len(variables[subplot_idx])],
                        variables[subplot_idx] * self.valid_frame,
                        color='b', lw=1)

                # plot zero line
                ax.axhline(0, color='gray')

            # plot threshold bending_angle_diff
            if variable_names[subplot_idx] == \
                    'Head vector angular speed (' + degree_sign + '/s)':
                ax.axhline(np.rad2deg(
                    par['threshold_head_vector_angular_speed']),
                    color='r', ls='--')
                ax.axhline(-np.rad2deg(
                    par['threshold_head_vector_angular_speed']),
                           color='g', ls='--')

            # plot threshold max of back vector angular speed
            if variable_names[subplot_idx] == \
                    'Tail vector angular speed (' + degree_sign + '/s)':
                ax.axhline(np.rad2deg(
                    par['threshold_back_vector_angular_speed']),
                    color='b',
                    ls='--')

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + figure_name)
            plt.close()

    def figure_time_series_variables_all_in_one(
            self, par,
            figure_name='time_series_variables_all_in_one'):

        # add or (un)comment variables and respective variable names
        variables = [
            self.tail_speed_forward,
            self.bending_angle,
            self.head_vector_angular_speed,
            self.back_vector_angular_speed
        ]

        variable_names = [
            'Tail speed forward (mm/s)',
            'Bending angle (' + degree_sign + ')',
            'Head vector angular speed (' + degree_sign + '/s)',
            'Back vector angular speed (' + degree_sign + '/s)'
        ]

        color = ['b', 'r', 'g', 'm', 'c']
        lw = [1, 1, 1, 1, 1]
        alpha = [1, 1, .5, .5, .5]

        # figure settings
        fig = plt.figure(figsize=(2 * par['fig_width'], 1 * par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(left=0.08, right=0.95, hspace=0.5,
                   wspace=0.5, bottom=0.15, top=0.8)

        # first row
        ax = plt.subplot(gs1[0, 0])
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        plt.setp(ax, xlabel='Time (s)',
                 xlim=(self.time[0], self.time[-1]),
                 ylabel='', ylim=(-5, 5))

        # plot HCs
        colors = {-1: 'g', 1: 'r'}
        for start_idx, end_idx, direction in zip(
                self.HC_start_idx, self.HC_end_idx,
                self.HC_direction):
            ax.axvspan(self.time[start_idx], self.time[end_idx],
                       facecolor=colors[direction], alpha=0.2, lw=0)

        # plot foward step
        for idx in self.step_idx:
            ax.axvline(self.time[idx], lw=2, color='k', alpha=0.2)

        # plot variables
        for idx in range(len(variables)):
            # scale
            y = (variables[idx] - np.nanmean(variables[idx])) / \
                np.nanstd(variables[idx])

            # plot
            ax.plot(self.time[:len(y)], y,
                    lw=lw[idx], alpha=alpha[idx], color=color[idx],
                    label=variable_names[idx])

        # plot zero
        ax.axhline(0, color='gray')

        # legend
        leg = ax.legend(
            loc=[
                0,
                1.05],
            ncol=2,
            handlelength=2,
            fontsize=font_size)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + figure_name)
            plt.close()
