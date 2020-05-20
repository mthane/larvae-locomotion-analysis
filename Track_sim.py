
# ==============================================================================
# Track_sim class
# ==============================================================================
from Track import *
import pandas 
import numpy
class Track_sim(Track):
    def __init__(self, par=None):

        self.time = np.arange(par['start_time'],
                              par['end_time'] + float(par['dt']),
                              float(par['dt']))
        self.n_steps = len(self.time)

        # description
        self.experiment = 'some_experiment_sim'
        self.condition = 'some_condition_sim'
        self.trial = 'some_trial_sim'
        self.track_number = 0
        self.ok = True

        # absolute spine points
        self.spine = []
        for idx in range(12):
            self.spine.append(np.zeros((self.n_steps, 2)) * np.nan)

        # valid_frame (=1. if valid, =np.nan else)
        self.valid_frame = np.ones(len(self.time))

        # start condition for spine point 5
        self.spine[5][0, :] = np.array([
            60. * (np.random.rand() - 0.5),
            20. * (np.random.rand() - 0.5)])  # in mm

        # start angle
        angle_start = np.pi * 2. * (np.random.random() - 0.5)  # in rad

        # spine0to5 vector
        self.spine0to5 = np.zeros((self.n_steps, 2)) * np.nan
        self.spine0to5[0, :] = (np.cos(angle_start) * 0.5 * par['spine_length'],
                                np.sin(angle_start) * 0.5 * par['spine_length'])

        # spine5to11 vector
        self.spine5to11 = np.zeros((self.n_steps, 2)) * np.nan
        self.spine5to11[0, :] = (
            np.cos(angle_start) * 0.5 * par['spine_length'],
            np.sin(angle_start) * 0.5 * par['spine_length'])

        # state
        self.state = np.array(self.n_steps * [''], dtype='|S12')
        self.state[0] = 'run'

    @staticmethod
    def draw_positive_HC_angle(par):
        HC_angle = np.abs(par['HC_angles_sigma'] * np.random.randn())
        while HC_angle > np.deg2rad(160.):
            HC_angle = np.abs(par['HC_angles_sigma'] * np.random.randn())
        return HC_angle

    #
    #
    #    @staticmethod
    #    def compute_bias(par, condition, d_previous, d_current):
    #
    #        bias = par['exp'][condition][0] * (
    #            np.exp(-par['exp'][condition][1] * d_previous) -
    #            np.exp(-par['exp'][condition][1] * d_current)
    #            )
    #
    #        return bias
    #
    #

    def simulate(self, par):

        print '  - simulating single track'

        # step_clock
        step_clock = 0.

        for i in range(self.n_steps - 1):

            # for each i, just one action should be executed
            action_taken = False

            # state = forward stepping
            if self.state[i] == 'run':

                action_taken = True

                # move if in first part of mean_INS_interval:
                if step_clock <= par['mean_INS_interval'] / 2.:

                    # keep spine5to11
                    self.spine5to11[i + 1, :] = self.spine5to11[i, :]

                    # infinitesimal forward step: forward_vector
                    forward_vector = (
                        float(par['dt']) * 2. *
                        par['mean_INS_distance'] /
                        par['mean_INS_interval'] *
                        self.spine5to11[i + 1, :] /
                        np.sqrt(np.sum(self.spine5to11[i + 1, :] ** 2)))

                    # move spine point 5
                    self.spine[5][i + 1, :] = (
                        self.spine[5][i, :] + forward_vector)

                    # change direction spine0to5...
                    self.spine0to5[i + 1, :] = (
                        self.spine0to5[i, :] + forward_vector)

                    # ... and set to correct length
                    self.spine0to5[i + 1, :] *= (
                        0.5 * par['spine_length'] /
                        np.sqrt(np.sum(self.spine0to5[i + 1, :] ** 2)))

                    # while head bumps into edge,
                    # rotate and overwrite spine5to11 i+1
                    while (
                        np.sqrt(np.sum((self.spine[5][i + 1, :] +
                                        self.spine5to11[i + 1, :]) ** 2)) >=
                        par['radius_dish']
                    ):
                        angle_tmp = clockwise_angle_from_first_to_second_vector(
                            first_vector=self.spine5to11[i + 1, :],
                            second_vector=self.spine[5][i + 1, :]
                        )
                        self.spine5to11[i + 1, :] = rotate_vector_clockwise(
                            angle=-0.02 * np.sign(angle_tmp),
                            vector=self.spine5to11[i + 1, :]
                        )

                    # increment step_clock
                    step_clock += float(par['dt'])

                    self.state[i + 1] = 'run'

                # dont move if in second part of T_step
                else:

                    self.spine5to11[i + 1, :] = self.spine5to11[i, :]
                    self.spine0to5[i + 1, :] = self.spine0to5[i, :]
                    self.spine[5][i + 1, :] = self.spine[5][i, :]

                    # increment step_clock
                    step_clock += float(par['dt'])

                    self.state[i + 1] = 'run'

                    # reset clock at end of T_wave
                    if step_clock > par['mean_INS_interval']:
                        step_clock = 0.

                        # transition to HC with p(HC|step)
                        if np.random.random() <= par['p(HC|step)']:

                            # next state
                            self.state[i + 1] = np.random.choice(['HC_left',
                                                                  'HC_right'])

                            # draw HC_angle
                            HC_angle = self.draw_positive_HC_angle(par)

                        # else keep running
                        else:
                            self.state[i + 1] = 'run'

            # HC_left
            if self.state[i] == 'HC_left' and not action_taken:

                action_taken = True

                # rotate front vector
                self.spine5to11[i + 1, :] = rotate_vector_clockwise(
                    angle=-1. * par['angular_speed'] * float(par['dt']),
                    vector=self.spine5to11[i, :])

                # keep spine point 5 and spine0to5
                self.spine[5][i + 1, :] = self.spine[5][i, :]
                self.spine0to5[i + 1, :] = self.spine0to5[i, :]

                # quit HC if HC_angle is reached
                bending_angle = clockwise_angle_from_first_to_second_vector(
                    first_vector=self.spine0to5[i + 1, :],
                    second_vector=self.spine5to11[i + 1, :])
                if bending_angle <= -HC_angle:

                    # HC to other side at HC end?
                    if np.random.random() <= par['p(HC|HC)']:
                        self.state[i + 1] = 'HC_right'

                        HC_angle = self.draw_positive_HC_angle(par)

                    # else resume run
                    else:
                        self.state[i + 1] = 'run'

                # else keep HC
                else:
                    self.state[i + 1] = 'HC_left'

                # quite HC if head bumps into edge
                if (np.sqrt(np.sum((
                    self.spine[5][i + 1, :] +
                    self.spine5to11[i + 1, :]) ** 2)) >=
                        par['radius_dish']):
                    self.state[i + 1] = 'run'

            # HC_right
            if self.state[i] == 'HC_right' and not action_taken:

                action_taken = True

                # rotate front vector
                self.spine5to11[i + 1, :] = rotate_vector_clockwise(
                    angle=1. * par['angular_speed'] * float(par['dt']),
                    vector=self.spine5to11[i, :])

                # keep midpoint and back vector
                self.spine[5][i + 1, :] = self.spine[5][i, :]
                self.spine0to5[i + 1, :] = self.spine0to5[i, :]

                # quit HC if sing_angle is reached
                bending_angle = clockwise_angle_from_first_to_second_vector(
                    first_vector=self.spine0to5[i + 1, :],
                    second_vector=self.spine5to11[i + 1, :])
                if bending_angle >= HC_angle:

                    # HC to other side at HC end?
                    if np.random.random() <= par['p(HC|HC)']:
                        self.state[i + 1] = 'HC_left'

                        HC_angle = self.draw_positive_HC_angle(par)

                    else:
                        self.state[i + 1] = 'run'

                # else keep HC
                else:
                    self.state[i + 1] = 'HC_right'

                # quite HC if head bumps into edge
                if (np.sqrt(np.sum((
                    self.spine[5][i + 1, :] +
                    self.spine5to11[i + 1, :]) ** 2)) >=
                        par['radius_dish']):
                    self.state[i + 1] = 'run'

        # check output
        if (self.state == '').any():
            print 'ERROR: missing state value!'

    def append_spine_and_contour(self, par):

        for i in range(0, 5):
            self.spine[i] = self.spine[5] - (5. - i) / 5. * self.spine0to5

        for i in range(6, 12):
            self.spine[i] = self.spine[5] + (i - 5.) / 6. * self.spine5to11

        # spine vectors
        self.spine_vectors = []
        for idx in range(11):
            self.spine_vectors.append(self.spine[idx + 1] - self.spine[idx])

        # contour points
        worm_width = np.array([0.28, 0.35, 0.4, 0.4, 0.4,
                               0.4, 0.4, 0.37, 0.3, 0.2])

        self.contour = []
        self.contour.append(self.spine[0])
        for i in range(10):
            orthogonal_vector = np.array(
                rotate_vector_clockwise(
                    np.pi / 2.,
                    [(self.spine_vectors[i] + self.spine_vectors[i + 1])
                     [:, j] for j in [0, 1]])).T
            self.contour.append(
                worm_width[i] *
                orthogonal_vector +
                self.spine[
                    i +
                    1])
        self.contour.append(self.spine[11])
        for i in range(9, -1, -1):
            orthogonal_vector = np.array(
                rotate_vector_clockwise(
                    np.pi / 2.,
                    [(self.spine_vectors[i] + self.spine_vectors[i + 1])
                     [:, j] for j in [0, 1]])).T
            self.contour.append(-
                                worm_width[i] *
                                orthogonal_vector +
                                self.spine[i +
                                           1])

        # delete unused attributes
        del self.spine0to5
        del self.spine5to11

    def filter_track_sim(self, par):

        # spine
        for i in range(12):
            for col in [0, 1]:
                self.spine[i][
                    :,
                    col] = np.convolve(
                        self.spine[i][
                            :,
                            col],
                        np.ones(
                            2 *
                            par['n_filter_points'] +
                            1) /
                        float(
                            2 *
                            par['n_filter_points'] +
                            1),
                        mode='same')
            self.spine[i] = self.spine[i][
                par['n_filter_points']: -par['n_filter_points'], :]

        # spine vectors
        for i in range(11):
            for col in [0, 1]:
                self.spine_vectors[i][
                    :,
                    col] = np.convolve(
                        self.spine_vectors[i][
                            :,
                            col],
                        np.ones(
                            2 *
                            par['n_filter_points'] +
                            1) /
                        float(
                            2 *
                            par['n_filter_points'] +
                            1),
                        mode='same')
            self.spine_vectors[i] = self.spine_vectors[i][
                par['n_filter_points']: -par['n_filter_points'], :]

        # contour
        for i in range(22):
            for col in [0, 1]:
                self.contour[i][
                    :,
                    col] = np.convolve(
                        self.contour[i][
                            :,
                            col],
                        np.ones(
                            2 *
                            par['n_filter_points'] +
                            1) /
                        float(
                            2 *
                            par['n_filter_points'] +
                            1),
                        mode='same')
            self.contour[i] = self.contour[i][
                par['n_filter_points']: -par['n_filter_points'], :]

        # correct time, state, valid_frame
        self.time = self.time[par['n_filter_points']: -par['n_filter_points']]
        self.state = self.state[par['n_filter_points']: -par['n_filter_points']]
        self.valid_frame = self.valid_frame[
            par['n_filter_points']: -
            par['n_filter_points']]


