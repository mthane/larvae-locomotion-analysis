import operator
import configparser as ConfigParser
import numpy as np
import os
import pickle as cPickle
import pandas
from Track import Track
from scipy.signal import argrelmax
from MiscFunctions import MiscFunctions as mf

class Tracks:
    '''collection of metadata for a given experiment and single condition,
        and respective figure functions'''

    def __init__(self, par=None, which_trials='all', which_tracks='all'):

        print('\n' + 50 * '=' + '\n\nLoading data from database...')
        # experiment and condition
        self.par = par
        self.which_trials = which_trials
        self.which_tracks = which_tracks
        self.experiment = par['experiment_name']
        self.group = par['group_name']
        group = self.group
        self.condition = par['condition']
        # self.dict = {}
        full_condition = par['group_name'] + '_-_' + par['condition']
        self.full_condition = par['group_name'] + '_-_' + par['condition']
        self.names = {
            full_condition: full_condition}

        self.names_short = {
            full_condition: full_condition}

        self.lc = {
            full_condition: 'r'}

        self.lw = {
            full_condition: 1.0}

        self.alpha = {
            full_condition: 0.12}

        # tmpdir = par['parent_dir'] + '/' + par['experiment_name']
        # + '/' + 'tmp'
        # tmpdir = (par['parent_dir'] + '/' +
        #          par['experiment_name'] + '/' +
        #          par['tracked_on'] + '/' + 'tmp')
        # print "============================"
        # print par['experiment_name']
        # print par['group_name']
        # print par['condition']
        # print "============================"

        if par['save_data']:
            self.excelWriter = pandas.ExcelWriter(
                par['data_dir'] +"multiple",
           
                #par['data_dir'] +
                #par['group_name'] +
                #'__' + par['condition'] +
                #'.xlsx',
             
                engine='xlsxwriter')

        # keys to collect
        keys = [

            # time-series variables
            'time', 'distance', 'trial', 'trial_number',
            'track_number', 'spine4',
            'bearing_angle',
            'heading_angle',
            'HC_initiation', 'tail_speed_forward',
            'head_speed_forward', 'bending_angle',
            'centroid_speed', 'midpoint_speed',
            'head_vector_angular_speed',
            'back_vector_angular_speed',
            'duration',

            # step variables
            'mean_INS_interval',
            'mean_INS_distance', 'mean_spine_length',
            'step_idx', 'step_idx_false',
            'INS_turn', 'next_event_is_step',
            'INS_distance', 'INS_interval',

            # HC variables
            'max_back_vector_angular_speed',
            'max_back_vector_angular_speed_false',
            'n_steps_in_HC', 'n_steps_in_HC_false',
            'HC_angle', 'HC_angle_false',
            'HC_start_idx', 'step_HC_interval', 'HC_step_interval', 'HC',
            'HC_end_idx'
        ]

        # init keys
        for key in keys:

            # 1dim keys
            if key in ['time', 'distance', 'trial', 'trial_number',
                       'track_number',
                       'mean_INS_interval',
                       'mean_INS_distance', 'mean_spine_length',
                       'tail_speed_forward', 'step_idx',
                       'step_idx_false', 'head_speed_forward',
                       'head_vector_angular_speed',
                       'back_vector_angular_speed',
                       'max_back_vector_angular_speed', 'midpoint_speed',
                       'centroid_speed',
                       'max_back_vector_angular_speed_false',
                       'n_steps_in_HC', 'n_steps_in_HC_false',
                       'HC_angle', 'HC_angle_false',
                       'INS_turn', 'HC_start_idx', 'next_event_is_step',
                       'INS_distance', 'INS_interval',
                       'bearing_angle', 'HC_initiation',
                       'heading_angle', 'HC_end_idx',
                       'step_HC_interval', 'HC_step_interval',
                       'duration', 'bending_angle', 'HC', 'area_averages']:
                setattr(self, key, np.array([]))

            # spine4
            if key == 'spine4':
                self.spine4 = np.array([[], []]).reshape(0, 2)

        print('\n - parent_dir: ', par['parent_dir'])
        print('\n - experiment: ', self.experiment)
        print('\n - group:', group)
        print('\n - condition:', full_condition)

        dir_condition = (
                par['parent_dir'] + '/' + self.experiment +
                '/' + par['tracked_on'] + '/' + group + '/' + self.condition).replace('/','',1)
        print(dir_condition)
        # for which trials?
        if which_trials == 'all':
            trials = [f for f in os.listdir(dir_condition) if
                      os.path.isdir(os.path.join(dir_condition, f)) and
                      f[0] != '.' and
                      len(
                          [c for c in os.listdir(os.path.join(dir_condition, f))
                           if c.endswith('.csv')]) != 0]
            print("TRIALS FOUND: ")
            print(trials)

        if which_trials == 'first':
            trials = [os.listdir(dir_condition)[0]]
        if isinstance(which_trials, list):
            trials = np.copy(which_trials)

        self.trials = trials

        # idx count for idx type
        idx_count = 0
        self.odor_A = np.array([])
        # for all trials
        # if(os.path.isfile(par['parent_dir'] + '/' + self.experiment +'/damaged_trials.txt')):
        #		os.remove(par['parent_dir'] + '/' + self.experiment +'/damaged_trials.txt')
        # damaged_trials = open(par['parent_dir'] + '/' + self.experiment +'/damaged_trials.txt',"w+")

        for trial_num, trial in enumerate(trials):

            print('\n   - trial:', trial)
            # Check for metadata.txt
            dir_trial = '/'.join((dir_condition, trial))

            print('\n   - trial dir:', dir_trial)

            trial_metadata = os.path.join(dir_trial, 'vidAndLogs/metadata.txt')

            if os.path.isfile(trial_metadata):
                trial_config = ConfigParser.RawConfigParser()
                trial_config.read(trial_metadata)
            try:
                par['odor_A'] = np.array(
                    list(map(float,
                             trial_config.get(
                                 'Trial Data', 'OdorALocation').split(','))))
            except:
                par['odor_A'] = None

            try:
                par['odor_B'] = np.array(
                    list(map(float,
                             trial_config.get(
                                 'Trial Data', 'OdorBLocation').split(','))))
            except:
                par['odor_B'] = None

            if (par['odor_B'] != None and par['odor_B'] != None):
                par['odor_A'] = [0.0, 0.0]
                par['odor_B'] = [0.0, 0.0]

            # else:
            print(par['odor_A'])
            print("Changing odor to: " + str(par['odor_A']))
            #if (par['odor_A'] != None):
            par['odor_A'] = [0.0, np.linalg.norm(par['odor_A'])]
            #else:
            #    par['odor_A'] = [0.0, 0.0]
            print("Changed odor to: " + str(par['odor_A']))

            if par['odor_A'] is not None and self.odor_A.size == 0:
                self.odor_A = np.hstack((self.odor_A, np.array(par['odor_A'])))
            elif par['odor_A'] is not None:
                self.odor_A = np.vstack((self.odor_A, np.array(par['odor_A'])))

            all_track_names = [x for x in os.listdir(dir_trial) if
                               par['path_suffix'] in x]
            # all_track_csv = [x for x in os.listdir(dir_trial) if '.csv' in x]
            if len(all_track_names) < 2 or par['rebuild'] is True:
                print("Rebuilding data")
                Track.write_database(par,
                               self.experiment,
                               group, [self.condition], [trial], 'all')

                all_track_names = [x for x in os.listdir(dir_trial)
                                   if par['path_suffix'] in x]

            # for which tracks?
            if which_tracks == 'all':
                track_names = all_track_names
            if which_tracks == 'first':
                track_names = [all_track_names[0]]
            if isinstance(which_tracks, list):
                track_names = [
                    str(track_number) +
                    par['path_suffix'] for track_number in which_tracks]

            # for all track.pkl
            for track_name in track_names:

                print('     - track:', track_name)

                file_path = '/'.join((dir_trial, track_name))

                # load track.pkl
                some_file = open(file_path, 'rb'
                                 )
                track = cPickle.load(some_file)
                print(track.__dict__)

                if (par['start_time'] != track.par['start_time'] or
                        par['end_time'] != track.par['end_time']):
                    print(('WARNING: Track loaded from pkl has ' +
                           'different duration than the current arguments'))
                    print('start_time: ' + str(par['start_time']))
                    print('end_time: ' + str(par['end_time']))
                    print('track.start_time: ' + str(track.par['start_time']))
                    print('track.end_time: ' + str(track.par['end_time']))
                # track = pickle.load(some_file)

                # check again
                track.check_analyzed_track(par)

                # if track is ok, collect data
                if track.ok:

                    # for all keys
                    for key in keys:

                        if key in ['time', 'distance',
                                   'mean_INS_interval',
                                   'mean_INS_distance',
                                   'mean_spine_length',
                                   'tail_speed_forward',
                                   'head_speed_forward',
                                   'head_vector_angular_speed',
                                   'back_vector_angular_speed',
                                   'centroid_speed',
                                   'midpoint_speed',
                                   'max_back_vector_angular_speed',
                                   'max_back_vector_angular_speed_false',
                                   'n_steps_in_HC', 'n_steps_in_HC_false',
                                   'HC_angle', 'HC_angle_false',
                                   'INS_turn', 'next_event_is_step',
                                   'INS_distance', 'INS_interval',
                                   'bearing_angle', 'HC_initiation',
                                   'heading_angle',
                                   'step_HC_interval', 'HC_step_interval',
                                   'duration', 'bending_angle', 'HC']:
                            setattr(self, key, np.hstack([getattr(self, key),
                                                          getattr(track, key)]))
                        if key in ['time']:
                            setattr(self, key + 'v',
                                    [getattr(self, key),
                                     list(getattr(track, key))])
                        # correct idx arrays
                        if key in ['step_idx', 'step_idx_false',
                                   'HC_start_idx', 'HC_end_idx']:
                            setattr(
                                self, key, np.hstack(
                                    [getattr(self, key), getattr(track, key) +
                                     idx_count]))

                        # add artificial time-series arrays for single values
                        if key in ['trial', 'track_number']:
                            setattr(self, key, np.hstack(
                                [getattr(self, key), np.array(
                                    len(track.time) *
                                    [getattr(track, key)])]))

                        # trial_number, add artificial time-series array
                        if key == 'trial_number':
                            setattr(self, key, np.hstack(
                                [getattr(self, key),
                                 trial_num * np.ones(len(track.time))]))

                        if key == 'spine4':
                            self.spine4 = np.vstack(
                                [self.spine4, track.spine[4]])

                    # increment idx_count
                    idx_count += len(track.time)

                # close file
                some_file.close()

        # convert to bool
        self.next_event_is_step = self.next_event_is_step.astype(bool)

        # convert to int
        self.trial_number = self.trial_number.astype(int)

        # convert idx type to int
        self.step_idx = self.step_idx.astype(int)
        self.step_idx_false = self.step_idx_false.astype(int)
        self.HC_start_idx = self.HC_start_idx.astype(int)
        self.HC_end_idx = self.HC_end_idx.astype(int)

        # self.dict[full_condition] = self

        print('\n...done')

    def save(self, par):
        # tmpdir = par['parent_dir'] + '/' + par['experiment_name']
        # + '/' + 'tmp'
        tmpdir = (par['parent_dir'] + '/' +
                  par['experiment_name'] + '/' +
                  par['tracked_on'] + '/' + 'tmp')
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        # full_condition = par['group_name'] + '-' + par['condition']
        path = (tmpdir + '/' + self.experiment +
                '_' +
                self.group + '_-_' +
                self.condition +
                par['path_suffix'])
        self.excelWriter = None
        print('Saving Tracks Pickle: ' + path)
        mf.save_pkl(
            self, path)

    def movie_tracks_on_dish(
            self,
            par,
            figure_name='tracks_on_dish'):
        # Time Step in secs
        dt = 1
        time_range = np.array([0, dt])
        # figure settings
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(
            left=0.02,
            right=0.98,
            hspace=0.1,
            wspace=0.1,
            bottom=0.02,
            top=0.98)
        gs1.get_subplot_params(fig)
        ax1 = plt.subplot(gs1[0, 0])
        ims = []
        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-
                            par['radius_dish'] -
                            5, par['radius_dish'] +
                            5), ylim=(-par['radius_dish'] -
                                      5, par['radius_dish'] +
                                      5), xticks=[], yticks=[])

        # plot edge of petri dish
        patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
                      fc='k', lw=0, alpha=0.1)
        ax1.add_artist(patch)

        n_bins = 100
        filter_points = 10
        xx, yy = np.meshgrid(
            np.linspace(-par['radius_dish'], par['radius_dish'], n_bins),
            np.linspace(-par['radius_dish'], par['radius_dish'], n_bins)
        )
        r_mask = np.ones((n_bins, n_bins))
        r_mask[
            np.sqrt(xx ** 2 + yy ** 2) >
            par['radius_dish'] - filter_points /
            2.0 / float(n_bins) * (2. * par['radius_dish'])] = np.nan
        end_time = np.max(self.time)
        for i in range(0, int(np.max(self.time) / dt) + 1):
            time_range = time_range + (dt)
            # time_range[1] = time_range[1] + (dt)
            restr_spine4 = self.spine4[
                (self.time >= time_range[0]) & (self.time < time_range[1])]

            sys.stdout.write("\r%3d/%d" % (i, end_time / dt))
            sys.stdout.flush()
            # print self.spine4.shape
            # print self.spine4[:100]
            # print restr_spine4.shape
            # print restr_spine4[:100]

            heatmap_mat, xedges, yedges = np.histogram2d(
                x=restr_spine4[~np.isnan(restr_spine4[:, 0]), 0],
                y=restr_spine4[~np.isnan(restr_spine4[:, 0]), 1],
                range=np.array(
                    [[-par['radius_dish'] - 5, par['radius_dish'] + 5],
                     [-par['radius_dish'] - 5, par['radius_dish'] + 5]]),
                bins=n_bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            # convolve and normalize
            #  heatmap_mat = convolve2d(
            #     in1=heatmap_mat,
            #     in2=np.ones((filter_points, filter_points)),
            #     mode='same', boundary='fill', fillvalue=0)

            heatmap_mat = ndimage.convolve(
                input=heatmap_mat,
                weights=np.ones((filter_points, filter_points)),
                output=None, mode='constant', cval=0.0, origin=0
            )
            heatmap_mat = ndimage.convolve(
                input=heatmap_mat,
                weights=np.ones((filter_points, filter_points)),
                output=None, mode='constant', cval=0.0, origin=0
            )

            heatmap_mat /= np.sum(heatmap_mat)

            im = ax1.imshow(
                heatmap_mat.T *
                r_mask,
                extent=extent,
                interpolation='nearest',
                cmap=plt.get_cmap("afmhot"),
                origin='lower')
            tx = plt.text(par['radius_dish'] - 10,
                          par['radius_dish'],
                          "%3ds" % i)
            ims.append([im, tx])
        # plot odor
        odor_A = np.mean(self.odor_A, axis=0)
        # print "=============== ODOR A POSITION ================"
        # print self.odor_A
        # print odor_A
        # if par['odor_A'] is not None:
        if odor_A is not None:
            # Dish will be rotated so we only need [0,magnitude]
            # for the correct position
            # mag = np.linalg.norm(par['odor_A'])
            mag = np.linalg.norm(odor_A)
            ax1.plot(
                0.0,
                mag,
                ls='none',
                color='w',
                marker='+',
                ms=10,
                mec='w',
                mfc='none',
                mew=1)
        ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)
        ani.save(par['figure_dir'] +
                 '/' + par['group_name'] + '_'
                                           '_' + par['condition'] + '_heatmap.mkv')

    def norm_individual_area_to_tracklength(
            self,
            par,
            trials,
            figure_name='tracks_on_dish'):
        # figure settings

        # plot edge of petri dish
        # patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
        #              fc='k', lw=0, alpha=0.1)
        # ax1.add_artist(patch)
        # plot petri dish in red
        # patch = Circle((.0, .0), par['radius_dish'], fc=(1.0, 0, 0), lw=0.0)
        # ax1.add_artist(patch)
        # plot spine4 trajectories
        # for all trials
        for trial in trials:

            # select trial
            trial_idx = self.trial == trial

            norm_black_sum = 0.0
            # for all tracks
            trial_tracks = np.unique(self.track_number[trial_idx])
            for track_number in trial_tracks:
                # start_time = timeit.default_timer()
                fig = plt.figure(
                    figsize=(
                        par['fig_width'],
                        par['fig_width']))
                gs1 = gridspec.GridSpec(1, 1)
                gs1.update(
                    left=0,
                    right=1.0,
                    hspace=0.,
                    wspace=0.,
                    bottom=0.,
                    top=1.0)
                gs1.get_subplot_params(fig)
                ax1 = plt.subplot(gs1[0, 0])
                [ax1.spines[str_tmp].set_color('none')
                 for str_tmp in ['top', 'right', 'left', 'bottom']]
                plt.setp(ax1,
                         xlim=(-par['radius_dish'],
                               par['radius_dish']),
                         ylim=(-par['radius_dish'],
                               par['radius_dish']),
                         xticks=[], yticks=[])
                patch = Circle((.0, .0), par['radius_dish'], fc=(1.0, 0, 0),
                               lw=0.0)
                ax1.add_artist(patch)
                idx_track = self.track_number[trial_idx] == track_number
                # Plot track
                ax1.plot(self.spine4[trial_idx, 0][idx_track],
                         self.spine4[trial_idx, 1][idx_track],
                         lw=par['area_line_width'], ls='-', color=(0, 0, 0))
                track_length = np.sum(
                    np.linalg.norm(
                        np.diff(self.spine4[trial_idx][idx_track],
                                axis=0),
                        axis=1))
                buf = io.BytesIO()
                # Save figure in memory
                # plt.show()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                im = Image.open(buf)
                black = 0.0
                # red = 0.0
                # Count black pixels
                for pixel in im.getdata():
                    if pixel == (0, 0, 0, 255):
                        black += 1
                buf.close()
                w, h = im.size
                pixel_mm = 2 * par['radius_dish'] / w
                area = black * pixel_mm * pixel_mm
                norm_black_sum += area / (1.0 * track_length)
        norm_black_average = norm_black_sum / (1.0 * len(trial_tracks))
        print("NORM AVERAGE = " + str(norm_black_average))
        return norm_black_average

    def group_area_trial_tracks_on_dish(
            self,
            par,
            trials,
            figure_name='tracks_on_dish'):
        # figure settings
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(
            left=0.02,
            right=0.98,
            hspace=0.1,
            wspace=0.1,
            bottom=0.02,
            top=0.98)
        gs1.get_subplot_params(fig)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-
                            par['radius_dish'] -
                            5, par['radius_dish'] +
                            5), ylim=(-par['radius_dish'] -
                                      5, par['radius_dish'] +
                                      5), xticks=[], yticks=[])

        # plot edge of petri dish
        # patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
        #              fc='k', lw=0, alpha=0.1)
        # ax1.add_artist(patch)
        # plot petri dish in red
        patch = Circle((.0, .0), par['radius_dish'], fc=(1.0, 0, 0), lw=0.0)
        ax1.add_artist(patch)
        # plot spine4 trajectories
        # for all trials
        for trial in trials:

            # select trial
            trial_idx = self.trial == trial

            # for all tracks
            for track_number in np.unique(self.track_number[trial_idx]):
                idx_track = self.track_number[trial_idx] == track_number
                ax1.plot(self.spine4[trial_idx, 0][idx_track],
                         self.spine4[trial_idx, 1][idx_track],
                         lw=par['area_line_width'], ls='-', color=(0, 0, 0))

        buf = io.BytesIO()
        # plt.show()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        black = 0.0
        red = 0.0
        for pixel in im.getdata():
            if pixel == (0, 0, 0, 255):
                black += 1
            elif pixel == (255, 0, 0, 255):
                red += 1
        return (black * 1.0) / (1.0 * (black + red))
        buf.close()

    def figure_tracks_on_dish(
            self,
            par,
            heatmap=False,
            figure_name='tracks_on_dish'):
        # figure settings
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(
            left=0.02,
            right=0.98,
            hspace=0.1,
            wspace=0.1,
            bottom=0.02,
            top=0.98)
        gs1.get_subplot_params(fig)
        ax1 = plt.subplot(gs1[0, 0])
        [ax1.spines[str_tmp].set_color('none')
         for str_tmp in ['top', 'right', 'left', 'bottom']]
        plt.setp(ax1, xlim=(-
                            par['radius_dish'] -
                            5, par['radius_dish'] +
                            5), ylim=(-par['radius_dish'] -
                                      5, par['radius_dish'] +
                                      5), xticks=[], yticks=[])

        # plot edge of petri dish
        patch = Wedge((.0, .0), par['radius_dish'] + 3, 0, 360, width=3,
                      fc='k', lw=0, alpha=0.1)
        ax1.add_artist(patch)

        # heatmap
        if heatmap:
            n_bins = 500
            heatmap_mat, xedges, yedges = np.histogram2d(
                x=self.spine4[~np.isnan(self.spine4[:, 0]), 0],
                y=self.spine4[~np.isnan(self.spine4[:, 0]), 1],
                range=np.array([[-1.0 * par['radius_dish'], par['radius_dish']],
                                [-1.0 * par['radius_dish'], par['radius_dish']]]
                               ),
                bins=n_bins)

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            # convolve and normalize
            filter_points = 20
            heatmap_mat = ndimage.convolve(
                input=heatmap_mat,
                weights=np.ones((filter_points, filter_points)),
                output=None, mode='constant', cval=0.0, origin=0
            )
            heatmap_mat /= np.sum(heatmap_mat)

            # r_mask
            xx, yy = np.meshgrid(
                np.linspace(-par['radius_dish'], par['radius_dish'], n_bins),
                np.linspace(-par['radius_dish'], par['radius_dish'], n_bins)
            )
            r_mask = np.ones((n_bins, n_bins))
            r_mask[
                np.sqrt(xx ** 2 + yy ** 2) >
                par['radius_dish'] - filter_points /
                1.0 / float(n_bins) * (2. * par['radius_dish'])] = np.nan

            ax1.imshow(
                heatmap_mat.T *
                r_mask,
                extent=extent,
                interpolation='nearest',
                origin='lower')

        # plot spine4 trajectories
        else:

            # all trial names
            trials = np.unique(self.trial)

            # for all trials
            for trial in trials:

                # select trial
                trial_idx = self.trial == trial

                # for all tracks
                for track_number in np.unique(self.track_number[trial_idx]):
                    idx_track = self.track_number[trial_idx] == track_number
                    ax1.plot(self.spine4[trial_idx, 0][idx_track],
                             self.spine4[trial_idx, 1][idx_track],
                             lw=1, ls='-', color='k', alpha=0.05)

        # plot odor
        odor_A = np.mean(self.odor_A, axis=0)
        # print "=============== ODOR A POSITION ================"
        # print self.odor_A
        # print odor_A
        # if par['odor_A'] is not None:
        if odor_A is not None:
            # Dish will be rotated so we only need [0,magnitude]
            # for the correct position
            # mag = np.linalg.norm(par['odor_A'])
            mag = np.linalg.norm(odor_A)
            ax1.plot(
                0.0,
                mag,
                ls='none',
                color='w',
                marker='+',
                ms=10,
                mec='w',
                mfc='none',
                mew=1)

        # save plot
        if heatmap:
            hmstr = 'heatmap'
        else:
            hmstr = ''

        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'tracks_on_dish' + '_' + hmstr)
            plt.close()

    def figure_density_depending_on_distance_and_time(
            self,
            par,
            figure_name='density_depending_on_distance_and_time'):

        # parameters
        edges_time = np.arange(par['start_time'], par['end_time'] + 1)
        edges_distance = np.arange(0, par['radius_dish'] * 2)

        # filter width
        n_distance = (2 * par['radius_dish']) / 15
        n_time = (par['end_time'] - par['start_time']) / 15

        density, xedges, yedges = np.histogram2d(
            x=self.distance[~np.isnan(self.distance)], y=self.time
            [~np.isnan(self.distance)], bins=(edges_distance, edges_time))
        # convolve2d
        density = convolve2d(
            density,
            np.ones(
                (int(n_distance),
                 n_time)),
            mode='same',
            boundary='fill',
            fillvalue=np.nan)

        # print density
        # normalize
        idx_non_zero = (np.nansum(density, 0) != 0)
        density[:, idx_non_zero] /= np.nansum(density[:, idx_non_zero], 0)

        # figure layout
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.get_subplot_params(fig)
        gs1.update(
            left=0.2,
            right=0.9,
            hspace=0.,
            wspace=0.,
            bottom=0.15,
            top=0.95)
        ax = plt.subplot(gs1[0, 0])

        # figure settings
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # xticks = np.array([0, 60, 120, 180, 240, 300])
        # xticklabels = range(0, 6)
        xticks = np.linspace(self.par['start_time'],
                             self.par['end_time'], 5)
        # xticklabels = range(0, 7)
        # yticks = np.array([0, 25, 50, 75])
        yticks = np.linspace(0., 2 * par['radius_dish'], 4)
        plt.setp(ax, xticks=xticks, yticks=yticks, xlabel='Time (sec)',
                 xlim=(self.par['start_time'], self.par['end_time']),
                 ylim=(0, 2 * par['radius_dish']),
                 ylabel='Distance (mm)')

        # print xedges
        # print yedges
        # plot
        cax = ax.imshow(density, interpolation='nearest',
                        aspect='auto',
                        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
                        cmap='jet', origin='lower')

        # colorbar
        if True:
            # cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.1)
            fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.1)
            ax.annotate('Density', xy=(1.0, .5),
                        xycoords='axes fraction',
                        size=font_size, horizontalalignment='left',
                        verticalalignment='center', rotation=90)

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'density_on_distance_and_time'
                        )
            plt.close()

    def figure_time_resolved_PREF(self, par, figure_name='time_resolved_PREF'):

        # parameters
        bin_edges = np.arange(par['start_time'], par['end_time'], par['dt'])
        # bin_edges = np.arange(par['start_time'], par['end_time'])  # in sec
        # print "BIN_EDGES: " + str(bin_edges)

        # init
        pref = []
        # group = par['group']
        # condition = par['condition']
        # full_condition
        # for all trials
        # TODO: Check if condition needs changes here
        # full_condition = self.full_condition
        line_data = np.array(bin_edges)[:-1]
        for trial_number in np.unique(self.trial_number):
            # init
            time_up = 0.
            time_down = 0.

            # select trial and not nan idx
            trial_idx = self.trial_number == trial_number
            not_nan_idx = ~np.isnan(self.spine4[:, 0])

            # time and spine4_y
            time_tmp = self.time[not_nan_idx * trial_idx]
            spine4_y = self.spine4[not_nan_idx * trial_idx, 1]

            # time up and down
            time_down = np.histogram(time_tmp[spine4_y < 0.], bin_edges)[0]
            # print "TIME_DOWN at Indices: " + str(time_tmp[spine4_y < 0.])
            # print "Hist output (Down): " + str(time_down)
            time_up = np.histogram(time_tmp[spine4_y > 0.], bin_edges)[0]
            # replace 1 each time the sum is zero. Sum zero means both time_up
            # and time_down are 0 (they are always positive)
            S = (time_up + time_down).astype(float)
            S[S == 0] = 1.0
            pref.append((time_up - time_down) / S)

        # convert to array
        pref = np.array(pref)

        line_data = np.vstack((line_data, np.median(pref, axis=0)))
        # boxplot lines
        boxplot_lines = np.percentile(pref, [10, 25, 50, 75, 90], axis=0)

        # figure
        fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))
        fig.subplots_adjust(
            left=0.2,
            right=0.9,
            hspace=0.,
            wspace=0.,
            bottom=0.15,
            top=0.95
        )
        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        plt.setp(
            ax, xlabel='Time (s)', ylabel='PREF', ylim=(-1.0, 1.0),
            xlim=(self.par['start_time'], self.par['end_time']),
            xticks=np.linspace(self.par['start_time'],
                               self.par['end_time'], 5)
        )
        # xlim=(0, 300),
        # xticks=[0, 100, 200, 300])

        # plot zero
        ax.axhline(0, color='gray')

        # plot median
        ax.plot(bin_edges[:-1], boxplot_lines[2], 'b-', lw=1)

        # plot quartiles and whiskers
        for idx, color in zip([4, 3, 2, 1], ['r', 'b', 'b', 'r']):
            ax.fill_between(bin_edges[:-1], boxplot_lines[idx],
                            boxplot_lines[idx - 1],
                            facecolor=color,
                            alpha=0.2, lw=0, edgecolor='None')

        # fake for legend
        ax.plot([100, 100], [100, 100], 'b-', alpha=1, lw=1, label='Median')
        ax.plot([100, 100], [100, 100], 'b-', alpha=0.2, lw=6, label='[25-75]')
        ax.plot([100, 100], [100, 100], 'r-', alpha=0.2, lw=6,
                label='[10-25[, ]75-90]')

        # legend
        leg = ax.legend(loc=4, ncol=1, handlelength=2, fontsize=font_size)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)
        # save data
        if par['save_data']:
            column_names = list([self.full_condition])
            column_names.insert(0, 'Time (s)')
            df = pandas.DataFrame(line_data.T)
            df.columns = column_names
            df.to_excel(self.excelWriter,
                        sheet_name='time_resolved_PREF',
                        index=False,
                        engine='xlsxwriter')
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'time_resolved_PREF'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'time_resolved_PREF'
                        )
            plt.close()

    def figure_track_numbers_per_time(self, par):
        ytime = np.arange(self.par['start_time'], self.par['end_time'] +
                          1.0 / par['fps'],
                          step=1.0 / par['fps'])
        idx_not_nan = ~np.isnan(self.time)
        track_counts_time = np.histogram(
            self.time[idx_not_nan], ytime,
            weights=np.ones(
                self.time[idx_not_nan].shape) / float(len(self.trials)))
        # save data
        if par['save_data']:
            df = pandas.DataFrame.from_records(track_counts_time).transpose()
            df.columns = ["Time",
                          "Average Number of Tracks"
                          ]
            df.to_excel(self.excelWriter,
                        sheet_name='track_counts_in_time',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'track_counts_in_time'])

    def figure_hist(self, par, key, figure_name='some_hist_plot'):

        # default parameters
        x_false = None
        vline = None
        n_bins = 100

        if key == 'duration':
            x = self.duration
            xlabel = 'Track duration (s)'
            xlim = (0, self.par['end_time'] - self.par['start_time'])
            xticks = np.linspace(0,
                                 self.par['end_time'] - self.par['start_time'],
                                 5)
            # xlim = (0, 300)
            # xticks = range(0, 301, 100)

        if key == 'tail_speed_forward':
            x = self.tail_speed_forward[self.step_idx]
            x_false = self.tail_speed_forward[self.step_idx_false]
            xlabel = 'Tail speed (mm/s)'
            xlim = (0, 3)
            xticks = list(range(0, 4))
            vline = par['threshold_tail_speed_forward']

        if key == 'HC_angle':
            x = np.rad2deg(self.HC_angle)
            xlabel = 'HC angle (' + degree_sign + ')'
            xlim = (-120, 120)
            xticks = list(range(-120, 121, 120))

        if key == 'Abs_HC_angle':
            x = np.rad2deg(np.abs(self.HC_angle))
            xlabel = 'Absolute HC angle (' + degree_sign + ')'
            xlim = (0, 120)
            xticks = list(range(0, 121, 120))

        if key == 'INS_interval':
            x = self.INS_interval
            x = x[~np.isnan(x)]
            xlabel = 'Inter-step-interval (s)'
            xlim = (0, 4)
            xticks = list(range(5))
            print('\n' + key \
                  + '\nMedian inter-step-interval = %.2f' % np.median(x))
            print('\n' + key \
                  + '\nMean inter-step-interval = %.2f' % np.mean(x))

        if key == 'INS_interval_no_HCs':
            x = self.INS_interval[self.next_event_is_step]
            xlabel = 'Inter-step-interval (s)'
            xlim = (0, 4)
            xticks = list(range(5))
            print('\n' + key \
                  + '\nMedian inter-step-interval = %.2f' % np.median(x))

        if key == 'back_vector_angular_speed':
            x = np.rad2deg(self.max_back_vector_angular_speed)
            x_false = np.rad2deg(self.max_back_vector_angular_speed_false)
            xlabel = 'Back ang. speed (' + degree_sign + '/s)'
            xlim = (0, 100)
            xticks = list(range(0, 101, 50))
            vline = np.rad2deg(par['threshold_back_vector_angular_speed'])

        # make hist
        edges = np.linspace(xlim[0], xlim[1], n_bins)
        hist = np.histogram(a=x, range=[np.nanmin(x), np.nanmax(x)],
                            bins=edges, density=False)[0]
        if x_false is not None:
            hist_false = np.histogram(x_false, bins=edges, density=False)[0]

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(
            left=0.2,
            right=0.9,
            hspace=0.,
            wspace=0.,
            bottom=0.15,
            top=0.95
        )
        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_major_locator(MaxNLocator(3))
        plt.setp(ax, xlabel=xlabel, ylabel='Count', xlim=xlim, xticks=xticks)

        # plot histogram x
        ax.bar(
            left=edges[
                 :-
                 1],
            height=hist,
            width=edges[1] - edges[0],
            bottom=0.,
            color='b',
            alpha=0.6,
            edgecolor='none',
            linewidth=0,
            align='edge')

        # plot histogram x_false
        if x_false is not None:
            ax.bar(
                left=edges[:-1],
                height=hist_false,
                width=edges[1] - edges[0],
                bottom=0.,
                color='k',
                alpha=0.4,
                edgecolor='none',
                linewidth=0,
                align='edge')

        # plot vline
        if vline is not None:
            ax.axvline(vline, color='blue', ls='--')

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'histogram' + str(key)
                        )
            plt.close()

    def figure_scatter_plot_with_hist(
            self,
            par,
            key,
            figure_name='some_scatter_plot'):

        # default parameters
        xlim = None
        ylim = None
        xlabel = ''
        ylabel = ''
        # fit_line = False
        hline = None
        vline = None
        x_false = None
        y_false = None

        # mean spine length vs mean mean_INS_distance
        if key == 'mean_spine_length_vs_mean_INS_distance':
            x = self.mean_spine_length
            y = self.mean_INS_distance
            xlabel = 'Mean spine length (mm)'
            ylabel = 'Mean inter-step-distance (mm)'
            xlim = (2, 6)
            ylim = (0.4, 1.4)
            alpha = 0.05
            # fit_line = True
            idx = (~np.isnan(x)) * (~np.isnan(y))
            print('\n' + key \
                  + '\npearson R = %.3f, p-value = %.10f' \
                  % pearsonr(x[idx], y[idx]))

        # mean spine length vs mean mean_INS_interval
        if key == 'mean_spine_length_vs_mean_INS_interval':
            x = self.mean_spine_length
            y = self.mean_INS_interval
            xlabel = 'Mean spine length (mm)'
            ylabel = 'Mean inter-step-interval (s)'
            xlim = (2, 6)
            ylim = (0.4, 1.4)
            alpha = 0.05
            # fit_line = True
            idx = (~np.isnan(x)) * (~np.isnan(y))
            print('\n' + key + '\npearson R = %.3f, p-value = %.10f' \
                  % pearsonr(x[idx], y[idx]))

        # spine length vs mean_INS_distance
        if key == 'INS_interval_vs_INS_distance':
            x = self.INS_interval
            y = self.INS_distance
            xlabel = 'Inter-step-interval (s)'
            ylabel = 'Inter-step-distance (mm)'
            xlim = (0, 4)
            ylim = (0, 4)
            alpha = 0.01
            print(('\n' + key + '\nInter-step-distance > 1.4mm = ' + str(
                np.sum(y[~np.isnan(y)] > 1.4) / float(len(y[~np.isnan(y)])))))

        # tail vs head speed for step detection
        if key == 'tail_speed_forward_vs_head_speed_forward':
            x = self.tail_speed_forward[self.step_idx]
            x_false = self.tail_speed_forward[self.step_idx_false]
            xlabel = 'Tail speed forward (mm/s)'
            y = self.head_speed_forward[self.step_idx]
            y_false = self.head_speed_forward[self.step_idx_false]
            ylabel = 'Head speed forward (mm/s)'
            vline = par['threshold_tail_speed_forward']
            alpha = 0.01
            xlim = (-0.2, 3.2)
            ylim = (-4, 4)
            print('\nMedian tail speed forward = %.3f' % np.median(x))

        # HC clusters, tail angular speed vs number of steps
        if key == 'tail_angular_speed_vs_n_steps_in_HC':
            x = np.rad2deg(self.max_back_vector_angular_speed)
            x_false = np.rad2deg(self.max_back_vector_angular_speed_false)
            xlabel = 'Back vector angular speed (' + degree_sign + '/s)'
            y = self.n_steps_in_HC
            y_false = self.n_steps_in_HC_false
            ylabel = 'Number of steps in HC'
            xlim = (0, 120)
            ylim = (-0.2, 3.2)
            vline = np.rad2deg(par['threshold_back_vector_angular_speed'])
            hline = par['threshold_n_steps_per_HC'] + 0.2
            alpha = 0.01

        # HC clusters, tail angular speed vs HC angle
        if key == 'tail_angular_speed_vs_HC_angle':
            x = np.rad2deg(self.max_back_vector_angular_speed)
            x_false = np.rad2deg(self.max_back_vector_angular_speed_false)
            xlabel = 'Back vector angular speed (' + degree_sign + '/s)'
            y = np.rad2deg(np.abs(self.HC_angle))
            y_false = np.rad2deg(np.abs(self.HC_angle_false))
            ylabel = '|HC angle (' + degree_sign + ')|'
            alpha = 0.02
            xlim = (0, 120)
            ylim = (0, 120)
            vline = np.rad2deg(par['threshold_back_vector_angular_speed'])

        # HC_time_to_last_step_vs_next_step
        if key == 'step_HC_interval_vs_HC_step_interval':
            x = self.step_HC_interval
            y = self.HC_step_interval
            xlabel = 'Step-HC-interval (s)'
            ylabel = 'HC-step-interval (s)'
            alpha = 0.01
            xlim = (0, 4)
            ylim = (0, 4)
            idx = (~np.isnan(x)) * (~np.isnan(y))
            print('\nMedian step-HC-interval = %.3f' % np.median(x[idx]))
            print('Median HC-step-interval = %.3f' % np.median(y[idx]))

        # bending angle vs inter-step-turn
        if key == 'bending_angle_vs_INS_turn':
            y = np.rad2deg(self.INS_turn)
            x = np.rad2deg(self.bending_angle[self.step_idx])
            idx = (~np.isnan(x)) * (~np.isnan(y))
            ylabel = 'Inter-step-turn (' + degree_sign + ')'
            xlabel = 'Bending angle (' + degree_sign + ')'
            xlim = (-90, 90)
            ylim = (-90, 90)
            alpha = 0.02
            print('\n' + key + '\npearson R = %.3f, p-value = %.10f' \
                  % pearsonr(x[idx], y[idx]))

        # step_HC_interval_vs_HC_angle
        if key == 'step_HC_interval_vs_HC_angle':
            x = self.step_HC_interval
            y = np.rad2deg(np.abs(self.HC_angle))
            xlabel = 'Step-HC-interval (s)'
            ylabel = '|HC angle| (' + degree_sign + ')'
            alpha = 0.01
            xlim = (0, 4)
            ylim = (0, 120)

        # no labels
        nullfmt = NullFormatter()

        # figure settings
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))

        print(str(fig))
        # definitions for the axes
        left = 0.2
        width = 0.6
        bottom = 0.15
        height = 0.53
        bottom_h = left_h = left + width + 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        ax = plt.axes(rect_scatter)
        ax_x = plt.axes(rect_histx)
        ax_y = plt.axes(rect_histy)

        ax_x.spines['top'].set_color('none')
        ax_x.spines['left'].set_color('none')
        ax_x.spines['right'].set_color('none')
        ax_x.xaxis.set_ticks_position('bottom')
        ax_x.yaxis.set_tick_params(width=0)

        ax_y.spines['top'].set_color('none')
        ax_y.spines['bottom'].set_color('none')
        ax_y.spines['right'].set_color('none')
        ax_y.yaxis.set_ticks_position('left')
        ax_y.xaxis.set_tick_params(width=0)

        # first plot
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        plt.setp(ax, xlabel=xlabel, ylabel=ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax_x.xaxis.set_major_locator(MaxNLocator(4))
        ax_x.yaxis.set_major_locator(MaxNLocator(1))

        ax_y.xaxis.set_major_locator(MaxNLocator(1))
        ax_y.yaxis.set_major_locator(MaxNLocator(4))

        plt.setp(ax_x, xlim=xlim)
        plt.setp(ax_y, ylim=ylim)

        # no labels
        ax_x.xaxis.set_major_formatter(nullfmt)
        ax_x.yaxis.set_major_formatter(nullfmt)
        ax_y.xaxis.set_major_formatter(nullfmt)
        ax_y.yaxis.set_major_formatter(nullfmt)

        # plot true
        ax.plot(
            x,
            y,
            mec=(
                0,
                0,
                1,
                alpha),
            ls='',
            marker='x',
            ms=3,
            mfc='None',
            mew=1)

        # plot false
        if x_false is not None:
            ax.plot(
                x_false,
                y_false,
                mec=(
                    1,
                    0,
                    0,
                    alpha),
                ls='',
                marker='x',
                ms=3,
                mfc='None',
                mew=1)

        # plot vline
        if vline is not None:
            ax.axvline(vline, color='b')

        # plot hline
        if hline is not None:
            ax.axhline(hline, color='b')

        # plot histogram x
        edges = np.linspace(xlim[0], xlim[1], 100)
        hist = np.histogram(x, bins=edges, density=False)[0]
        ax_x.bar(
            left=edges[:-1],
            height=hist,
            width=edges[1] - edges[0],
            bottom=0.,
            color='b',
            alpha=0.6,
            edgecolor='none',
            linewidth=0,
            align='edge')

        hist_false = np.histogram(x_false, bins=edges, density=False)[0]
        ax_x.bar(left=edges[:-1], height=hist_false, width=edges[1] - edges[0],
                 bottom=hist, color='r', alpha=0.6,
                 edgecolor='none', linewidth=0, align='edge')

        # plot histogram y
        edges = np.linspace(ylim[0], ylim[1], 100)
        hist = np.histogram(y, bins=edges, density=False)[0]
        ax_y.barh(bottom=edges[:-1], width=hist, height=edges[1] - edges[0],
                  left=0., color='b',
                  ec='none', linewidth=0, alpha=0.6)

        hist_false = np.histogram(y_false, bins=edges, density=False)[0]
        ax_y.barh(
            bottom=edges[:-1],
            width=hist_false,
            height=edges[1] - edges[0],
            left=hist,
            color='r',
            ec='none',
            linewidth=0,
            alpha=0.6)

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'scatter_plot_' + str(key)
                        )
            plt.close()

    def figure_3d_scatter_plot(self, par, key, figure_name='3d_scatter_plot'):

        if key == 'step_HC_interval_vs_HC_step_interval_vs_HC_angle':

            x = self.HC_step_interval
            y = self.step_HC_interval
            z = np.abs(np.rad2deg(self.HC_angle))

            # figure layout
            fig = plt.figure(
                figsize=(
                    1.5 *
                    par['fig_width'],
                    1.5 *
                    par['fig_width']))
            ax = fig.add_subplot(111, projection='3d')

            # figure settings
            plt.setp(
                ax, xticks=list(range(4)), yticks=list(range(4)), zticks=list(range(0, 61, 30)),
                xlabel='Step-HC-interval (s)', ylabel='HC-step-interval (s)',
                zlabel='|HC angle| (' + degree_sign + ')', xlim=(-0.1, 3.1),
                ylim=(-0.1, 3.1), zlim=(-5, 65))

            # plot
            ax.scatter(x, y, z, color='r', marker='x', alpha=0.2)

            # save plot
            if par['save_figure']:
                plt.savefig(par['figure_dir'] +
                            '/' + figure_name)
            plt.close()

    def figure_heatmap_distance_and_bearing_with_projection(
            self,
            par,
            variable_name,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # box filter
        box_half_points = 11
        box = np.ones((box_half_points * 2, box_half_points * 2))
        dstr = ""
        dstrshort = ""
        # mean_INS_distance, mean_INS_interval, step_turning_angle
        if variable_name in ['INS_distance',
                             'INS_interval', 'INS_turn']:
            distance = self.distance[self.step_idx[self.next_event_is_step]]
            bearing_angle = self.bearing_angle[
                self.step_idx[
                    self.next_event_is_step]]
            weights = getattr(self, variable_name)[self.next_event_is_step]

            if variable_name == 'INS_turn':
                weights = np.rad2deg(weights)

        # HC rate
        if variable_name == 'HC_rate':
            idx_not_nan = ~np.isnan(self.HC_initiation)
            distance = self.distance[idx_not_nan]
            bearing_angle = self.bearing_angle[idx_not_nan]
            weights = self.HC_initiation[idx_not_nan] / float(par['dt'])

        # HC angle
        if variable_name == 'HC_angle':
            if subthreshold:
                dstr = "(max angle " + str(large_HC) + ")"
                dstrshort = "_maxA" + str(int(large_HC))
            else:
                dstr = "(min angle " + str(large_HC) + ")"
                dstrshort = "_minA" + str(int(large_HC))
            # large_HC_idx = np.abs(self.HC_angle) > np.deg2rad(
            #     par['large_HC'])
            large_HC_idx = angleComp(self.HC_angle, large_HC,
                                     subthreshold)
            distance = self.distance[self.HC_start_idx[large_HC_idx]]
            bearing_angle = self.bearing_angle[self.HC_start_idx[large_HC_idx]]
            weights = np.rad2deg(self.HC_angle[large_HC_idx])

        # number of samples in each bin
        n_samples = np.histogram2d(
            x=distance,
            y=bearing_angle,
            bins=(
                par['edges_distance'],
                par['edges_bearing']),
            normed=False)[0]

        # number of samples in each rectangle
        n_samples = convolve2d(n_samples, box, mode='same',
                               boundary='wrap', fillvalue=np.nan)
        n_samples[n_samples == 0] = np.nan

        # hist2d
        hist2d = np.histogram2d(
            x=distance,
            y=bearing_angle,
            bins=(
                par['edges_distance'],
                par['edges_bearing']),
            normed=False,
            weights=weights)[0]

        # convolve hist2d
        hist2d = convolve2d(hist2d, box, mode='same',
                            boundary='wrap', fillvalue=np.nan)

        # correct heatmap
        heatmap = hist2d / n_samples
        heatmap[:box_half_points, :] = np.nan
        heatmap[-box_half_points:, :] = np.nan
        heatmap[n_samples < 15] = np.nan

        # color limits
        # vmin_vmax = {
        #    'INS_interval': (0.7, 0.9),
        #    'INS_turn': (0.0, 3.0),
        #    'INS_distance': (0.7, 1.0),
        #    'HC_rate': (0.14, 0.27),
        #    'HC_angle': (-12, 12)
        #    }

        color_bar_title = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')'
        }

        # x and y ticks and ticklabels
        bearing_tick_labels = np.array([-180., -90., 0., 90., 180.])
        # distance_tick_labels = np.array([0., 50., 100, 150.])
        # distance_tick_labels = np.array([0., 20., 40, 60., 80.])
        distance_tick_labels = np.linspace(0., 2 * par['radius_dish'], 4)
        bearing_ticks = [(np.abs(par['edges_bearing'] - i)).argmin()
                         for i in np.pi / 180. * bearing_tick_labels]
        distance_ticks = [
            (np.abs(par['edges_distance'] - i)).argmin
            () for i in distance_tick_labels]

        # figure layout
        fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))

        # definitions for the axes
        left, width = 0.19, 0.4
        bottom, height = 0.19, 0.4
        bottom_h = left_h = left + width + 0.1

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width + 0.19, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        ax = plt.axes(rect_scatter)
        ax_x = plt.axes(rect_histx)
        ax_y = plt.axes(rect_histy)

        # figure settings
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax_x.spines['top'].set_color('none')
        ax_x.spines['right'].set_color('none')
        ax_x.spines['left'].set_color('r')
        ax_x.spines['bottom'].set_color('r')
        ax_x.xaxis.set_ticks_position('bottom')
        ax_x.yaxis.set_ticks_position('left')
        ax_x.tick_params(axis='x', colors='red')
        ax_x.tick_params(axis='y', colors='red')
        ax_x.yaxis.label.set_color('r')
        ax_x.xaxis.label.set_color('r')

        ax_y.spines['top'].set_color('none')
        ax_y.spines['right'].set_color('none')
        ax_y.spines['bottom'].set_color('b')
        ax_y.spines['left'].set_color('b')
        ax_y.xaxis.set_ticks_position('bottom')
        ax_y.yaxis.set_ticks_position('left')
        ax_y.tick_params(axis='x', colors='b')
        ax_y.tick_params(axis='y', colors='b')
        ax_y.yaxis.label.set_color('b')
        ax_y.xaxis.label.set_color('b')

        plt.setp(
            ax,
            xlabel='Distance (mm)',
            ylabel='Bearing angle (' + degree_sign + ')',
            xticks=distance_ticks,
            xticklabels=np.array(
                distance_tick_labels).astype(int),
            yticks=bearing_ticks,
            yticklabels=np.array(bearing_tick_labels).astype(int),
            xlim=[
                distance_ticks[0],
                distance_ticks[-1]],
            ylim=[bearing_ticks[0],
                  bearing_ticks[-1]]
        )

        ticks = np.nanmean(np.abs(heatmap.T), 0)
        ticks = [np.nanmin(ticks), np.nanmax(ticks)]
        plt.setp(ax_x, xticks=distance_ticks, xlim=(0, heatmap.shape[0]),
                 yticks=ticks,
                 xticklabels=[],
                 yticklabels=[str(np.round(tick, 2)) for tick in ticks])

        ticks = np.nanmean(heatmap.T, 1)
        ticks = [np.nanmin(ticks), np.nanmax(ticks)]
        plt.setp(
            ax_y,
            ylim=(
                0,
                heatmap.shape[0]),
            yticks=bearing_ticks,
            yticklabels=[],
            xticks=ticks,
            xticklabels=[
                str(
                    np.round(
                        tick,
                        2)) for tick in ticks])

        # imshow
        cax = ax.imshow(
            heatmap.T,
            interpolation='nearest',
            aspect='auto',
            cmap='jet',
            origin='lower')
        #    vmin=vmin_vmax[variable_name][0],
        #    vmax=vmin_vmax[variable_name][1])

        # colorbar
        # cbar = fig.colorbar(cax,
        fig.colorbar(cax,
                     ax=ax_x,
                     orientation='vertical',
                     pad=0.17,
                     aspect=5,
                     ticks=[np.nanmin(heatmap),
                            np.nanmin(heatmap) + (np.nanmax(heatmap) -
                                                  np.nanmin(heatmap)) / 2.,
                            np.nanmax(heatmap)])

        ax.annotate(
            color_bar_title[variable_name] + dstr, xy=(.97, 0.97),
            xycoords='figure fraction', size=font_size,
            horizontalalignment='right', verticalalignment='top', rotation=0)

        # plot projections
        ax_x.plot(np.arange(heatmap.shape[0]), np.nanmean(np.abs(heatmap.T), 0),
                  'r-')
        ax_y.plot(np.nanmean(heatmap.T, 1), np.arange(heatmap.shape[0]),
                  'b-')

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'heatmap_distance_and_bearing_with_projection_' +
                        str(variable_name) + dstrshort
                        )
            plt.close()
            plt.close()

    def figure_heatmap_distance_and_bearing(self, par, variable_name,
                                            figure_name='heatmap_tmp',
                                            subthreshold=False,
                                            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # box filter
        box_half_points = 11
        box = np.ones((box_half_points * 2, box_half_points * 2))

        dstr = ""
        dstrshort = ""

        # mean_INS_distance, mean_INS_interval, step_turning_angle
        if variable_name in ['INS_distance', 'INS_interval',
                             'INS_turn']:
            distance = self.distance[self.step_idx[self.next_event_is_step]]
            bearing_angle = self.bearing_angle[
                self.step_idx[
                    self.next_event_is_step]]
            weights = getattr(self, variable_name)[self.next_event_is_step]

            if variable_name == 'INS_turn':
                weights = np.rad2deg(weights)

            # delete very unrealistic outliers
            if variable_name == 'INS_distance':
                idx_ok = weights < 1.5  # must be smaller than 1.5 mm
                distance = distance[idx_ok]
                bearing_angle = bearing_angle[idx_ok]
                weights = weights[idx_ok]

            if variable_name == 'INS_interval':
                idx_ok = weights < 10  # must be smaller than 10 seconds
                distance = distance[idx_ok]
                bearing_angle = bearing_angle[idx_ok]
                weights = weights[idx_ok]

        # HC rate
        if variable_name == 'HC_rate':
            idx_not_nan = ~np.isnan(self.HC_initiation)
            distance = self.distance[idx_not_nan]
            bearing_angle = self.bearing_angle[idx_not_nan]
            weights = self.HC_initiation[idx_not_nan] / float(par['dt'])

        # HC angle
        if variable_name in ['HC_angle']:
            if subthreshold:
                dstr = "(max angle " + str(large_HC) + ")"
                dstrshort = "_maxA" + str(int(large_HC))
            else:
                dstr = "(min angle " + str(large_HC) + ")"
                dstrshort = "_minA" + str(int(large_HC))
            large_HC_idx = angleComp(self.HC_angle, large_HC,
                                     subthreshold)
            distance = self.distance[self.HC_start_idx[large_HC_idx]]
            bearing_angle = self.bearing_angle[self.HC_start_idx[large_HC_idx]]
            weights = np.rad2deg(self.HC_angle[large_HC_idx])

        # Absolute HC angle
        if variable_name in ['Abs_HC_angle']:
            if subthreshold:
                dstr = "(max angle " + str(large_HC) + ")"
                dstrshort = "_maxA" + str(int(large_HC))
            else:
                dstr = "(min angle " + str(large_HC) + ")"
                dstrshort = "_minA" + str(int(large_HC))
            large_HC_idx = angleComp(self.HC_angle, large_HC,
                                     subthreshold)
            distance = self.distance[self.HC_start_idx[large_HC_idx]]
            bearing_angle = self.bearing_angle[self.HC_start_idx[large_HC_idx]]
            weights = np.abs(np.rad2deg(self.HC_angle[large_HC_idx]))

        # Run Speed
        if variable_name == 'run_speed':
            idx_not_nan = ~np.isnan(self.midpoint_speed)
            idx_non_hc = self.HC == 0
            # Leave some distance before and after HC
            idx_non_hc = np.invert(np.convolve(
                np.invert(idx_non_hc),
                (par['gap'] * 2 + 1) * [1], mode='same') > 0)

            idx_non_hc = idx_non_hc * idx_not_nan
            distance = self.distance[idx_non_hc]
            bearing_angle = self.bearing_angle[idx_non_hc]
            weights = self.midpoint_speed[
                idx_non_hc]

        # number of samples in each bin
        n_samples = np.histogram2d(
            x=distance,
            y=bearing_angle,
            bins=(
                par['edges_distance'],
                par['edges_bearing']),
            normed=False)[0]

        # number of samples in each rectangle
        n_samples = convolve2d(n_samples, box, mode='same',
                               boundary='wrap', fillvalue=np.nan)
        n_samples[n_samples == 0] = np.nan

        # hist2d
        hist2d = np.histogram2d(
            x=distance,
            y=bearing_angle,
            bins=(
                par['edges_distance'],
                par['edges_bearing']),
            normed=False,
            weights=weights)[0]

        # convolve hist2d
        hist2d = convolve2d(hist2d, box, mode='same',
                            boundary='wrap', fillvalue=np.nan)

        # correct heatmap
        heatmap = hist2d / n_samples
        heatmap[:box_half_points, :] = np.nan
        heatmap[-box_half_points:, :] = np.nan
        heatmap[n_samples < 15] = np.nan

        # color limits
        # vmin_vmax = {
        #     'INS_interval': (0.75, 0.82),
        #     'INS_turn': (-2, 2),
        #     'INS_distance': (0.7, 0.75),
        #     'HC_rate': (0.14, 0.27),
        #     'HC_angle': (-12, 12)
        #     }

        color_bar_title = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')',
            'Abs_HC_angle': 'Absolute HC angle (' + degree_sign + ')',
            'run_speed': 'Run Speed(mm/s)'
        }

        # x and y ticks and ticklabels
        bearing_tick_labels = np.array([-180., -90., 0., 90., 180.])
        # distance_tick_labels = np.array([0., 50., 100, 150.])
        # distance_tick_labels = np.array([0., 20., 40, 60., 80.])
        distance_tick_labels = np.linspace(0., 2 * par['radius_dish'], 4)
        bearing_ticks = [(np.abs(par['edges_bearing'] - i)).argmin()
                         for i in np.pi / 180. * bearing_tick_labels]
        distance_ticks = [
            (np.abs(par['edges_distance'] - i)).argmin
            () for i in distance_tick_labels]

        # save data
        if par['save_data']:
            dist = par['edges_distance'][:-1]
            bear = np.rad2deg(par['edges_bearing'][:-1])
            nheatmap = np.vstack((bear, heatmap.T))
            nheatmap = np.vstack((dist, nheatmap))
            nheatmap = nheatmap.T
            df = pandas.DataFrame(nheatmap)
            df.to_excel(self.excelWriter,
                        sheet_name='heatmap_' +
                                   str(variable_name) + dstrshort,
                        index=False)

        # figure layout
        fig = plt.figure(
            figsize=(
                0.7 *
                par['fig_width'],
                0.5 *
                par['fig_width']))
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(left=0.25, right=0.8, hspace=0., wspace=0., bottom=0.25,
                   top=0.75)
        ax = plt.subplot(gs1[0, 0])

        # figure settings
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.setp(
            ax, xlabel='Distance (mm)', ylabel='Bearing angle (' + degree_sign +
                                               ')', xticks=distance_ticks, xticklabels=np.array(
                distance_tick_labels).astype(int), yticks=bearing_ticks,
            yticklabels=np.array(bearing_tick_labels).astype(int), xlim=[
                distance_ticks[0], distance_ticks[-1]], ylim=[bearing_ticks[0],
                                                              bearing_ticks[-1]]
        )

        # imshow
        cax = ax.imshow(heatmap.T, interpolation='nearest', aspect='auto',
                        cmap='jet', origin='lower')  # ,
        #                vmin=vmin_vmax[variable_name][0],
        #                vmax=vmin_vmax[variable_name][1])

        # colorbar
        # cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.1,
        fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.1,
                     aspect=10)
        #             ticks=[vmin_vmax[variable_name][0],
        #                    (vmin_vmax[variable_name][0] +
        #                    vmin_vmax[variable_name][1])/2.,
        #                    vmin_vmax[variable_name][1]])

        # colorbar title
        ax.annotate(color_bar_title[variable_name] + dstr, xy=(1.7, 1.2),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        par['condition'] + '_' +
                        'heatmap_distance_and_bearing_' +
                        str(variable_name) +
                        dstrshort
                        )
            plt.close()

    def figure_HC_direction_serial_correlation(
            self,
            par,
            figure_name='HC_direction_serial_correlation'):

        not_nan_idx = ~np.isnan(self.HC_angle)
        HC_directions = np.array(
            len(self.HC_angle[not_nan_idx]) * ['left'], dtype='|S5')
        HC_directions[self.HC_angle[not_nan_idx] > 0] = 'right'

        probs = {}
        Ns = {}

        for previous_dir, current_dir in itertools.product(
                ['left', 'right'], ['left', 'right']):
            previous_HC = 1. * (HC_directions == previous_dir)[:-1]
            current_HC = 1. * (HC_directions == current_dir)[1:]
            probs['p(HC ' + current_dir + '|HC ' + previous_dir + ')'] = (
                    np.sum(previous_HC * current_HC) /
                    np.float(np.sum(previous_HC))
            )

            Ns['p(HC ' +
               current_dir +
               '|HC ' +
               previous_dir +
               ')'] = np.sum(previous_HC)

        for current_dir in ['left', 'right']:
            current_HC = 1. * (HC_directions == current_dir)
            probs['p(HC ' + current_dir + ')'] = (
                    np.sum(current_HC) /
                    np.float(len(current_HC))
            )

            Ns['p(HC ' + current_dir + ')'] = len(current_HC)

        print('')
        for key in list(probs.keys()):
            print(key, probs[key].round(3))

        prob_keys = [
            'p(HC left)', 'p(HC right)',
            'p(HC left|HC left)', 'p(HC right|HC left)',
            'p(HC left|HC right)', 'p(HC right|HC right)'
        ]

        # figure layout
        fig = plt.figure(figsize=(par['fig_width'], par['fig_width']))
        fig.subplots_adjust(left=0.2, right=.95, bottom=0.4, top=.9,
                            wspace=.2, hspace=.2)
        ax = fig.add_subplot(111)

        # settings
        plt.setp(
            ax, xlim=(-0.5, len(prob_keys) - 0.5), xticks=list(range(len(prob_keys))),
            ylim=(0.0, 1.0), ylabel='Probability')
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # plot bars
        ax.bar(list(range(len(prob_keys))), [probs[key] for key in prob_keys],
               width=0.6, color='b', alpha=0.5, edgecolor='none', linewidth=0,
               align='center')

        for i in range(len(prob_keys)):
            ax.annotate(
                'N=' + str(int(Ns[prob_keys[i]])), xy=(i / 6. + 0.08, .8),
                xycoords='axes fraction', size=font_size,
                horizontalalignment='center', verticalalignment='left',
                rotation=90)

        ax.set_xticklabels(prob_keys, rotation=45, ha='right', size=font_size)
        ax.axhline(y=0.5, color='k', alpha=0.5)

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_'
                                                  '_' + par['condition'] + '_' +
                        'HC_serial_correlation'
                        )
            plt.close()

    def figure_variable_depending_on_bearing_distance_split(
            self,
            par,
            variable_name, distance,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # figure settings
        ylabel = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')',
            'Abs_HC_angle': 'Absolute HC angle (' + degree_sign + ')',
            'run_speed': 'Run Speed(mm/s)'
        }
        # dstr = ""
        dstrshort = ""

        # ylim = {
        #     'INS_interval': (0.7, 0.8),
        #     'INS_turn': (-3, 3),
        #     'INS_distance': (0.7, 0.8),
        #     'HC_rate': (0.2, 0.35),
        #     'HC_angle': (-10, 10),
        #     'run_speed': (0.4, 0.5)
        #     }

        # yticks = {
        #     'INS_interval': [0.7, 0.75, 0.8],
        #     'INS_turn': [-3, 0, 3],
        #     'INS_distance': [0.7, 0.75, 0.8],
        #     'HC_rate': [0.2, 0.25, 0.3, 0.35],
        #     'HC_angle': [-10, -5, 0, 5, 10],
        #     'run_speed': [0.4, 0.45, 0.5]
        #     # 'HC_angle': [-5, -2.5, 0, 2.5, 5],
        #     }

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.15, top=0.85)

        # groups = [par['groups']]
        # conditions = [par['condition']]

        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        # line_data_far = np.rad2deg(edges_bearing)[:-1]
        # for all conditions

        # mean_INS_distance, mean_INS_interval,
        # step_turning_angle
        if variable_name in ['INS_distance', 'INS_interval',
                             'INS_turn']:

            near_next_event_step_idx = (
                np.where(self.distance[self.step_idx[
                    self.next_event_is_step]] < distance)[0])

            far_next_event_step_idx = (
                np.where(self.distance[self.step_idx[
                    self.next_event_is_step]] >= distance)[0])

            near_bearing_angle = self.bearing_angle[
                self.step_idx[near_next_event_step_idx]
            ]
            far_bearing_angle = self.bearing_angle[
                self.step_idx[far_next_event_step_idx]
            ]

            near_weights = getattr(
                self,
                variable_name)[
                near_next_event_step_idx]
            far_weights = getattr(
                self,
                variable_name)[
                far_next_event_step_idx]
            if variable_name == 'INS_turn':
                near_weights = np.rad2deg(near_weights)
                far_weights = np.rad2deg(far_weights)
                near_idx_ok = np.isnan(near_weights) == 0
                far_idx_ok = np.isnan(far_weights) == 0
                near_bearing_angle = near_bearing_angle[near_idx_ok]
                far_bearing_angle = far_bearing_angle[far_idx_ok]
                near_weights = near_weights[near_idx_ok]
                far_weights = far_weights[far_idx_ok]

            # delete very unrealistic outliers
            if variable_name == 'INS_distance':
                near_idx_ok = near_weights < 1.5  # smaller than 1.5 mm
                far_idx_ok = far_weights < 1.5  # smaller than 1.5 mm
                # add distance separation when needed
                near_bearing_angle = near_bearing_angle[near_idx_ok]
                far_bearing_angle = far_bearing_angle[far_idx_ok]
                near_weights = near_weights[near_idx_ok]
                far_weights = far_weights[far_idx_ok]

            if variable_name == 'INS_interval':
                near_idx_ok = near_weights < 10  # smaller than 10 seconds
                far_idx_ok = far_weights < 10  # smaller than 10 seconds
                near_bearing_angle = near_bearing_angle[near_idx_ok]
                far_bearing_angle = far_bearing_angle[near_idx_ok]
                near_weights = near_weights[near_idx_ok]
                far_weights = far_weights[far_idx_ok]

        # HC rate
        if variable_name == 'HC_rate':
            idx_not_nan = ~np.isnan(
                self.HC_initiation)
            near_idx_not_nan = (idx_not_nan & (self.distance < distance))
            far_idx_not_nan = (idx_not_nan & (self.distance >= distance))

            # print idx_not_nan.shape
            near_bearing_angle = self.bearing_angle[near_idx_not_nan]
            far_bearing_angle = self.bearing_angle[far_idx_not_nan]
            near_weights = self.HC_initiation[
                               near_idx_not_nan] / float(par['dt'])
            far_weights = self.HC_initiation[
                              far_idx_not_nan] / float(par['dt'])

        # Run Speed
        if variable_name == 'run_speed':
            idx_not_nan = ~np.isnan(self.midpoint_speed)
            idx_non_hc = self.HC == 0
            idx_non_hc = idx_non_hc * idx_not_nan
            near_idx_non_hc = (idx_non_hc & (self.distance < distance))
            far_idx_non_hc = (idx_non_hc & (self.distance >= distance))
            near_bearing_angle = self.bearing_angle[
                near_idx_non_hc]
            far_bearing_angle = self.bearing_angle[
                far_idx_non_hc]
            # weights = self.dict[self.full_condition].centroid_speed[
            #     idx_non_hc]
            near_weights = self.midpoint_speed[
                near_idx_non_hc]
            far_weights = self.midpoint_speed[
                far_idx_non_hc]

        # HC angle
        if variable_name in ['HC_angle']:
            near_HC_start_idx = self.HC_start_idx[
                self.distance[self.HC_start_idx] < distance]
            far_HC_start_idx = self.HC_start_idx[
                self.distance[self.HC_start_idx] >= distance]
            near_HC_angle = self.HC_angle[
                self.distance[self.HC_start_idx] < distance]
            far_HC_angle = self.HC_angle[
                self.distance[self.HC_start_idx] >= distance]
            if subthreshold:
                # dstr = "(max angle " + str(large_HC) + ")"
                dstrshort = "_maxA" + str(int(large_HC))
            else:
                # dstr = "(min angle " + str(large_HC) + ")"
                dstrshort = "_minA" + str(int(large_HC))
            near_large_HC_idx = angleComp(near_HC_angle, large_HC,
                                          subthreshold)
            far_large_HC_idx = angleComp(far_HC_angle, large_HC,
                                         subthreshold)
            near_bearing_angle = self.bearing_angle[
                near_HC_start_idx[near_large_HC_idx]]
            far_bearing_angle = self.bearing_angle[
                far_HC_start_idx[far_large_HC_idx]]
            near_weights = np.rad2deg(
                near_HC_angle[near_large_HC_idx])
            far_weights = np.rad2deg(
                far_HC_angle[far_large_HC_idx])

        # Absolute HC angle
        if variable_name in ['Abs_HC_angle']:
            near_HC_start_idx = self.HC_start_idx[
                self.distance[self.HC_start_idx] < distance]
            far_HC_start_idx = self.HC_start_idx[
                self.distance[self.HC_start_idx] >= distance]
            near_HC_angle = self.HC_angle[
                self.distance[self.HC_start_idx] < distance]
            far_HC_angle = self.HC_angle[
                self.distance[self.HC_start_idx] >= distance]
            if subthreshold:
                # dstr = "(max angle " + str(large_HC) + ")"
                dstrshort = "_maxA" + str(int(large_HC))
            else:
                # dstr = "(min angle " + str(large_HC) + ")"
                dstrshort = "_minA" + str(int(large_HC))
            near_large_HC_idx = angleComp(near_HC_angle, large_HC,
                                          subthreshold)
            far_large_HC_idx = angleComp(far_HC_angle, large_HC,
                                         subthreshold)
            near_bearing_angle = self.bearing_angle[
                near_HC_start_idx[near_large_HC_idx]]
            far_bearing_angle = self.bearing_angle[
                far_HC_start_idx[far_large_HC_idx]]
            near_weights = np.fabs(np.rad2deg(
                near_HC_angle[near_large_HC_idx]))
            far_weights = np.fabs(np.rad2deg(
                far_HC_angle[far_large_HC_idx]))

        # add data for circular boundary conditions
        near_bearing_angle = np.hstack(
            [near_bearing_angle - 2 *
             np.pi, near_bearing_angle, near_bearing_angle + 2 * np.pi])
        near_weights = np.tile(near_weights, 3)
        far_bearing_angle = np.hstack(
            [far_bearing_angle - 2 *
             np.pi, far_bearing_angle, far_bearing_angle + 2 * np.pi])
        far_weights = np.tile(far_weights, 3)

        # hist
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        near_n_samples = np.histogram(near_bearing_angle,
                                      bins=edges_bearing,
                                      normed=False)[0]
        far_n_samples = np.histogram(far_bearing_angle, bins=edges_bearing,
                                     normed=False)[0]
        near_hist = np.histogram(near_bearing_angle, bins=edges_bearing,
                                 normed=False, weights=near_weights)[0]
        far_hist = np.histogram(far_bearing_angle, bins=edges_bearing,
                                normed=False, weights=far_weights)[0]
        near_hist = near_hist / near_n_samples
        far_hist = far_hist / far_n_samples

        # convolve, filter width = 60 degree
        near_hist = np.convolve(np.ones(11) / 11., near_hist, mode='same')
        far_hist = np.convolve(np.ones(11) / 11., far_hist, mode='same')
        line_data = np.vstack((line_data, near_hist))
        line_data = np.vstack((line_data, far_hist))

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #        ax.yaxis.set_major_locator(MaxNLocator(3))
        ylimmin = np.min([np.min(near_hist), np.min(far_hist)])
        ylimmin = ylimmin - (0.1 * np.abs(ylimmin))
        ylimmax = np.max([np.max(near_hist), np.max(far_hist)])
        ylimmax = ylimmax + (0.1 * np.abs(ylimmax))
        plt.setp(ax, xlabel='Bearing angle (' + degree_sign + ')',
                 ylabel=ylabel[variable_name], xlim=(-180, 180),
                 ylim=(ylimmin, ylimmax), xticks=list(range(-180, 181, 90)))

        # vspan
        if variable_name in ['INS_distance', 'INS_interval',
                             'HC_rate', 'run_speed']:
            ax.axvspan(-180, -90, facecolor='k', alpha=0.15, lw=0)
            ax.axvspan(90, 180, facecolor='k', alpha=0.15, lw=0)

        if variable_name in ['INS_turn', 'HC_angle']:
            ax.axvspan(0, 180, facecolor='k', alpha=0.15, lw=0)

        # plot
        ax.plot(np.rad2deg(edges_bearing)[:-1], near_hist, ls='-',
                lw=self.lw[self.full_condition],
                color=self.lc[self.full_condition],
                alpha=2 * self.alpha[self.full_condition],
                label=self.names[self.full_condition] +
                      ' d<' + str(distance)
                )
        # plot
        ax.plot(np.rad2deg(edges_bearing)[:-1], far_hist, ls='-',
                lw=self.lw[self.full_condition],
                color=self.lc[self.full_condition],
                alpha=4 * self.alpha[self.full_condition],
                label=self.names[self.full_condition] +
                      ' d>=' + str(distance)
                )

        # legend
        leg = ax.legend(
            loc=[
                0,
                1.0],
            ncol=1,
            handlelength=1.5,
            fontsize=font_size)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)
        # save data
        if par['save_data']:
            column_names = [par['condition'] + '_d<' + str(distance),
                            par['condition'] + '_d>=' + str(distance)
                            ]
            column_names.insert(0, 'Bearing Angle')
            saved_line_idx = np.abs(line_data[0]) <= 183
            df = pandas.DataFrame(line_data.T[saved_line_idx])
            df.columns = column_names
            df.to_excel(self.excelWriter,
                        sheet_name=str(variable_name) +
                                   '_bear_line_d' + dstrshort,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    str(variable_name) +
                    '_bear_line_d' + dstrshort])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        str(variable_name) +
                        '_to_bearing_line_d' + dstrshort)
            plt.close()

    def figure_boxplot_variable_depending_on_bearing_distance_split(
            self,
            par,
            variable_name,
            distance,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # this function takes very long to compute, because of ...== trial
        # dstr = ""
        dstrshort = ""

        # figure settings
        ylabel = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')',
            'Abs_HC_angle': 'Absolute HC angle (' + degree_sign + ')',
            'run_speed': 'Run Speed(mm/s)'
        }

        column_names_short = {
            'INS_interval': ['toward', 'away'],
            'INS_turn': ['left', 'right'],
            'INS_distance': ['toward', 'away'],
            'HC_rate': ['toward', 'away'],
            'run_speed': ['toward', 'away'],
            'HC_angle': ['left', 'right'],
            'Abs_HC_angle': ['turn towards', 'turn away'],
        }
        tta = ['turn toward/near', 'turn toward/far',
               'turn away/near', 'turn away/far']
        ta = ['toward/near', 'toward/far', 'away/near', 'away/far']
        lr = ['left/near', 'left/far', 'right/near', 'right/far']
        column_names = {
            'INS_interval': ta,
            'INS_turn': lr,
            'INS_distance': ta,
            'HC_rate': ta,
            'run_speed': ta,
            'HC_angle': lr,
            'Abs_HC_angle': tta,
        }

        # ylim = {
        #     'INS_interval': (0.8, 0.9),
        #     'INS_turn': (-1.4, 1.4),
        #     'INS_distance': (0.7, 0.8),
        #     'HC_rate': (0.3, 0.7),
        #     'HC_angle': (-20, 20),
        #     }

        # yticks = {
        #     'INS_interval': [0.8, 0.85, 0.9],
        #     'INS_turn': [-1.4, 0, 1.4],
        #     'INS_distance': [0.7, 0.75, 0.8],
        #     'HC_rate': [0.3, 0.5, 0.7],
        #     'HC_angle': [-20, -10, 0, 10, 20],
        #     }

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot_black_near = []
        boxplot_white_near = []
        boxplot_black_far = []
        boxplot_white_far = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition
        # for all trials
        for trial_number in trial_numbers:

            # mean_INS_distance,
            # mean_INS_interval, step_turning_angle
            if variable_name in ['INS_distance',
                                 'INS_interval',
                                 'INS_turn']:

                near_next_event_step_idx = self.next_event_is_step[
                    self.distance[self.step_idx[
                        self.next_event_is_step]] < distance
                    ]
                far_next_event_step_idx = self.next_event_is_step[
                    self.distance[self.step_idx[
                        self.next_event_is_step]] >= distance
                    ]

                near_idx_trial = (
                        self.trial_number[
                            self.step_idx[
                                near_next_event_step_idx]] == trial_number)

                far_idx_trial = (
                        self.trial_number[
                            self.step_idx[
                                far_next_event_step_idx]] == trial_number)

                near_bearing_angle = self.bearing_angle[
                    self.step_idx[
                        near_next_event_step_idx]]
                near_bearing_angle = near_bearing_angle[near_idx_trial]

                far_bearing_angle = self.bearing_angle[
                    self.step_idx[
                        far_next_event_step_idx]]
                far_bearing_angle = far_bearing_angle[far_idx_trial]

                near_weights = getattr(self, variable_name)[
                    near_next_event_step_idx]
                near_weights = near_weights[near_idx_trial]

                far_weights = getattr(self, variable_name)[
                    far_next_event_step_idx]
                far_weights = far_weights[far_idx_trial]

                if variable_name == 'INS_turn':
                    near_weights = np.rad2deg(near_weights)
                    far_weights = np.rad2deg(far_weights)
                    near_idx_ok = np.isnan(near_weights) == 0
                    far_idx_ok = np.isnan(far_weights) == 0
                    near_bearing_angle = near_bearing_angle[near_idx_ok]
                    far_bearing_angle = far_bearing_angle[far_idx_ok]
                    near_weights = near_weights[near_idx_ok]
                    far_weights = far_weights[far_idx_ok]

                # delete very unrealistic outliers
                if variable_name == 'INS_distance':
                    near_idx_ok = near_weights < 1.5  # smaller than 1.5 mm
                    far_idx_ok = far_weights < 1.5  # smaller than 1.5 mm
                    # add distance separation when needed
                    near_bearing_angle = near_bearing_angle[near_idx_ok]
                    far_bearing_angle = far_bearing_angle[far_idx_ok]
                    near_weights = near_weights[near_idx_ok]
                    far_weights = far_weights[far_idx_ok]

                if variable_name == 'INS_interval':
                    near_idx_ok = near_weights < 10  # smaller than 10 seconds
                    far_idx_ok = far_weights < 10  # smaller than 10 seconds
                    near_bearing_angle = near_bearing_angle[near_idx_ok]
                    far_bearing_angle = far_bearing_angle[near_idx_ok]
                    near_weights = near_weights[near_idx_ok]
                    far_weights = far_weights[far_idx_ok]

            # HC rate
            if variable_name == 'HC_rate':
                idx_not_nan = ~np.isnan(
                    self.HC_initiation)
                near_idx_not_nan = (idx_not_nan & (self.distance < distance))
                far_idx_not_nan = (idx_not_nan & (self.distance >= distance))
                idx_trial = self.trial_number == trial_number
                near_idx_not_nan = near_idx_not_nan * idx_trial
                far_idx_not_nan = far_idx_not_nan * idx_trial
                # print idx_not_nan.shape
                near_bearing_angle = self.bearing_angle[near_idx_not_nan]
                far_bearing_angle = self.bearing_angle[far_idx_not_nan]
                near_weights = self.HC_initiation[
                                   near_idx_not_nan] / float(par['dt'])
                far_weights = self.HC_initiation[
                                  far_idx_not_nan] / float(par['dt'])

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(self.midpoint_speed)
                idx_non_hc = self.HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                idx_non_hc = idx_non_hc * idx_not_nan
                near_idx_non_hc = (idx_non_hc & (self.distance < distance))
                far_idx_non_hc = (idx_non_hc & (self.distance >= distance))
                idx_trial = self.trial_number == trial_number
                near_idx_non_hc = near_idx_non_hc * idx_trial
                far_idx_non_hc = far_idx_non_hc * idx_trial
                near_bearing_angle = self.bearing_angle[
                    near_idx_non_hc]
                far_bearing_angle = self.bearing_angle[
                    far_idx_non_hc]
                # weights = self.dict[self.full_condition].centroid_speed[
                #     idx_non_hc]
                near_weights = self.midpoint_speed[
                    near_idx_non_hc]
                far_weights = self.midpoint_speed[
                    far_idx_non_hc]

            # HC angle
            if variable_name in ['HC_angle', 'Abs_HC_angle']:
                near_HC_start_idx = self.HC_start_idx[
                    self.distance[self.HC_start_idx] < distance]
                far_HC_start_idx = self.HC_start_idx[
                    self.distance[self.HC_start_idx] >= distance]

                near_HC_angle = self.HC_angle[
                    self.distance[self.HC_start_idx] < distance]
                far_HC_angle = self.HC_angle[
                    self.distance[self.HC_start_idx] >= distance]

                if subthreshold:
                    # dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    # dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                near_large_HC_idx = angleComp(near_HC_angle, large_HC,
                                              subthreshold)
                far_large_HC_idx = angleComp(far_HC_angle, large_HC,
                                             subthreshold)

                near_idx_trial = (
                        self.trial_number
                        [near_HC_start_idx[near_large_HC_idx]] ==
                        trial_number)
                far_idx_trial = (
                        self.trial_number
                        [far_HC_start_idx[far_large_HC_idx]] ==
                        trial_number)

                near_bearing_angle = self.bearing_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_bearing_angle = self.bearing_angle[
                    far_HC_start_idx[far_large_HC_idx]]

                near_weights = np.rad2deg(
                    near_HC_angle[near_large_HC_idx])
                far_weights = np.rad2deg(
                    far_HC_angle[far_large_HC_idx])

                near_bearing_angle = near_bearing_angle[near_idx_trial]
                far_bearing_angle = far_bearing_angle[far_idx_trial]

                near_weights = near_weights[near_idx_trial]
                far_weights = far_weights[far_idx_trial]

                # DEBUG
                # print '==================HC ANGLE ==============='
                # print "trial number: " + str(trial_number)
                # print "near_HC_start_idx: " + str(near_HC_start_idx.shape)
                # print "far_HC_start_idx: " + str(far_HC_start_idx.shape)
                # print "near_HC_angle: " + str(near_HC_angle.shape)
                # print "far_HC_angle: " + str(far_HC_angle.shape)
                # print "near_large_HC_idx: " + str(near_large_HC_idx.shape)
                # print "far_large_HC_idx: " + str(far_large_HC_idx.shape)
                # print "near_idx_trial: " + str(near_idx_trial.shape)
                # print "far_idx_trial: " + str(far_idx_trial.shape)
                # print "near_bearing_angle: " + str(near_bearing_angle.shape)
                # print "far_bearing_angle: " + str(far_bearing_angle.shape)
                # print "near_weights: " + str(near_weights.shape)
                # print "far_weights: " + str(far_weights.shape)

            # apend boxplotdata
            if variable_name in ['INS_distance',
                                 'INS_interval']:
                idx_black_near = np.abs(near_bearing_angle) < par['to_range']
                idx_white_near = np.abs(near_bearing_angle) > par['away_range']

                idx_black_far = np.abs(far_bearing_angle) < par['to_range']
                idx_white_far = np.abs(far_bearing_angle) > par['away_range']

                boxplot_black_near.append(
                    np.mean(near_weights[idx_black_near]))
                boxplot_white_near.append(
                    np.mean(near_weights[idx_white_near]))

                boxplot_black_far.append(
                    np.mean(far_weights[idx_black_far]))
                boxplot_white_far.append(
                    np.mean(far_weights[idx_white_far]))

                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][0] + ' near',
                                     np.mean(near_weights[idx_black_near])])
                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][0] + ' far',
                                     np.mean(far_weights[idx_black_far])])
                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][1] + ' near',
                                     np.mean(near_weights[idx_white_near])])
                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][1] + ' far',
                                     np.mean(far_weights[idx_white_far])])

            if variable_name in ['HC_rate', 'run_speed']:
                near_idx_black = np.abs(near_bearing_angle) < par['to_range']
                near_idx_white = np.abs(near_bearing_angle) > par['away_range']
                far_idx_black = np.abs(far_bearing_angle) < par['to_range']
                far_idx_white = np.abs(far_bearing_angle) > par['away_range']
                boxplot_black_near.append(
                    np.sum(
                        near_weights[near_idx_black]) / len(
                        near_weights[near_idx_black]))
                boxplot_white_near.append(
                    np.sum(
                        near_weights[near_idx_white]) / len(
                        near_weights[near_idx_white]))

                boxplot_black_far.append(
                    np.sum(
                        far_weights[far_idx_black]) / len(
                        far_weights[far_idx_black]))
                boxplot_white_far.append(
                    np.sum(
                        far_weights[far_idx_white]) / len(
                        far_weights[far_idx_white]))

                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][0],
                    np.sum(near_weights[near_idx_black]) / len(
                        near_weights[near_idx_black])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][1],
                    np.sum(far_weights[far_idx_black]) / len(
                        far_weights[far_idx_black])
                ])

                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][2],
                    np.sum(near_weights[near_idx_white]) / len(
                        near_weights[near_idx_white])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][3],
                    np.sum(far_weights[far_idx_white]) / len(
                        far_weights[far_idx_white])
                ])

            if variable_name in ['INS_turn',
                                 'HC_angle', 'Abs_HC_angle']:
                if variable_name in ['Abs_HC_angle']:
                    # Towards
                    near_idx_black = near_bearing_angle * near_weights < 0.
                    # Away
                    near_idx_white = near_bearing_angle * near_weights > 0.
                    # Towards
                    far_idx_black = far_bearing_angle * far_weights < 0.
                    # Away
                    far_idx_white = far_bearing_angle * far_weights > 0.

                    near_weights = np.abs(near_weights)
                    far_weights = np.abs(far_weights)
                else:
                    near_idx_black = near_bearing_angle < 0.
                    near_idx_white = near_bearing_angle > 0.
                    far_idx_black = far_bearing_angle < 0.
                    far_idx_white = far_bearing_angle > 0.

                boxplot_black_near.append(np.mean(near_weights[near_idx_black]))
                boxplot_white_near.append(np.mean(near_weights[near_idx_white]))

                boxplot_black_far.append(np.mean(far_weights[far_idx_black]))
                boxplot_white_far.append(np.mean(far_weights[far_idx_white]))

                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][0],
                    np.mean(near_weights[near_idx_black])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][1],
                    np.mean(far_weights[far_idx_black])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][2],
                    np.mean(near_weights[near_idx_white])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' +
                    column_names[variable_name][3],
                    np.mean(far_weights[far_idx_white])
                ])

        # make black boxplot
        bp = ax.boxplot(boxplot_black_near, positions=[0],
                        widths=0.4, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        bp = ax.boxplot(boxplot_black_far, positions=[1],
                        widths=0.4, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # make white boxplot
        bp = ax.boxplot(boxplot_white_near, positions=[3],
                        widths=0.4, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')
        bp = ax.boxplot(boxplot_white_far, positions=[4],
                        widths=0.4, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel[variable_name],
            xlim=(-0.5, 4.5),
            xticks=[0, 1, 3, 4])
        ax.set_xticklabels(
            column_names[variable_name],
            rotation=45, ha='right', size=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        ax.annotate(par['condition'] + " (d=" + str(distance) + ")",
                    xy=(0.75, 1.0),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          '/'.join(column_names_short[variable_name]) +
                          " near/far (d=" + str(distance) + ")",
                          str(variable_name)]
            df.to_excel(self.excelWriter,
                        sheet_name=str(variable_name) +
                                   '_bear_box_d' + dstrshort,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    str(variable_name) + '_bear_box_d' + dstrshort])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        str(variable_name) + '_to_bearing_box_d' + dstrshort)
            plt.close()

    # def figure_boxplot_proportion_of_HC(
    #         self, par):
    #     ylabel = 'proportion of HCs'
    #     column_names = ['toward', 'away']

    #     # figure
    #     fig = plt.figure(
    #         figsize=(
    #             par['fig_width'],
    #             par['fig_width']))
    #     fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
    #                         bottom=0.25, top=0.9)

    #     # init
    #     boxplot_data = []
    #     boxplot_black = []
    #     boxplot_white = []
    #     pref_trials = np.unique(self.trial)
    #     trial_numbers = self.trial_number
    #     full_condition = self.full_condition
    #     # for all trials
    #     for trial_number in np.unique(trial_numbers):
    #         large_HC_idx = np.abs(
    #             self.HC_angle) > np.deg2rad(20.)
    #         print self.HC_angle
    #         bearing_angle = self.bearing_angle[
    #             self.HC_start_idx[large_HC_idx]]
    #         weights = np.rad2deg(
    #             self.HC_angle[large_HC_idx])
    #         bearing_angle = bearing_angle[idx_trial]
    #         weights = weights[idx_trial]
    #     multiplied = bearing_angle * weights

    def figure_variable_depending_on_bearing(self, par,
                                             variable_name,
                                             subthreshold=False,
                                             large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # dstr = ""
        dstrshort = ""

        # figure settings
        ylabel = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')',
            'Abs_HC_angle': 'Absolute HC angle (' + degree_sign + ')',
            'run_speed': 'Run Speed(mm/s)'
        }

        ylim = {
            'INS_interval': (0.7, 0.8),
            'INS_turn': (-3, 3),
            'INS_distance': (0.7, 0.8),
            'HC_rate': (0.2, 0.35),
            'HC_angle': (-10, 10),
            'Abs_HC_angle': (0, 10),
            'run_speed': (0.4, 0.5)
        }

        yticks = {
            'INS_interval': [0.7, 0.75, 0.8],
            'INS_turn': [-3, 0, 3],
            'INS_distance': [0.7, 0.75, 0.8],
            'HC_rate': [0.2, 0.25, 0.3, 0.35],
            'HC_angle': [-10, -5, 0, 5, 10],
            'Abs_HC_angle': [0, 2.5, 5, 7.5, 10],
            'run_speed': [0.4, 0.45, 0.5]
            # 'HC_angle': [-5, -2.5, 0, 2.5, 5],
        }

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.15, top=0.85)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #        ax.yaxis.set_major_locator(MaxNLocator(3))
        plt.setp(ax, xlabel='Bearing angle (' + degree_sign + ')',
                 ylabel=ylabel[variable_name], xlim=(-180, 180),
                 xticks=list(range(-180, 181, 90)), ylim=ylim[variable_name],
                 yticks=yticks[variable_name])

        # vspan
        if variable_name in ['INS_distance', 'INS_interval',
                             'HC_rate', 'run_speed']:
            ax.axvspan(-180, -90, facecolor='k', alpha=0.15, lw=0)
            ax.axvspan(90, 180, facecolor='k', alpha=0.15, lw=0)

        if variable_name in ['INS_turn', 'HC_angle']:
            ax.axvspan(0, 180, facecolor='k', alpha=0.15, lw=0)

        # groups = [par['groups']]
        conditions = [par['condition']]

        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        # for all conditions
        for condition in conditions:

            # mean_INS_distance, mean_INS_interval,
            # step_turning_angle
            if variable_name in ['INS_distance', 'INS_interval',
                                 'INS_turn']:

                bearing_angle = self.bearing_angle[
                    self.step_idx[
                        self.next_event_is_step]]

                weights = getattr(
                    self,
                    variable_name)[
                    self.next_event_is_step]

                if variable_name == 'INS_turn':
                    weights = np.rad2deg(weights)

                # delete very unrealistic outliers
                if variable_name == 'INS_distance':
                    idx_ok = weights < 1.5  # must be smaller than 1.5 mm
                    # add distance separation when needed
                    bearing_angle = bearing_angle[idx_ok]
                    weights = weights[idx_ok]

                if variable_name == 'INS_interval':
                    idx_ok = weights < 10  # must be smaller than 10 seconds
                    bearing_angle = bearing_angle[idx_ok]
                    weights = weights[idx_ok]

            # HC rate
            if variable_name == 'HC_rate':
                idx_not_nan = ~np.isnan(
                    self.HC_initiation)
                bearing_angle = self.bearing_angle[idx_not_nan]
                weights = self.HC_initiation[
                              idx_not_nan] / float(par['dt'])

            # HC rate
            # if variable_name == 'HC_rate':
            #     idx_not_nan = ~np.isnan(
            #         self.HC_initiation)
            #     print "=========HC_initiation==============="
            #     print self.HC_initiation.shape
            #     print "=========Distance==============="
            #     print self.distance.shape
            #     # print idx_not_nan.shape
            #     # TODO: FIX DISTANCE ADJUSTMENT
            #     idx_not_nan = (idx_not_nan & (self.distance < 69.0))
            #     # print idx_not_nan.shape
            #     bearing_angle = self.bearing_angle[idx_not_nan]
            #     weights = self.HC_initiation[
            #         idx_not_nan]/float(par['dt'])

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(self.midpoint_speed)
                idx_non_hc = self.HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                idx_non_hc = idx_non_hc * idx_not_nan
                bearing_angle = self.bearing_angle[
                    idx_non_hc]
                # weights = self.dict[self.full_condition].centroid_speed[
                #     idx_non_hc]
                weights = self.midpoint_speed[
                    idx_non_hc]

            # HC angle
            if variable_name == 'HC_angle':
                if subthreshold:
                    # dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    # dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(self.HC_angle, large_HC,
                                         subthreshold)
                bearing_angle = self.bearing_angle[
                    self.HC_start_idx[large_HC_idx]]
                weights = np.rad2deg(
                    self.HC_angle[large_HC_idx])

            # Abs HC angle
            if variable_name == 'Abs_HC_angle':
                if subthreshold:
                    # dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    # dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(self.HC_angle, large_HC,
                                         subthreshold)
                bearing_angle = self.bearing_angle[
                    self.HC_start_idx[large_HC_idx]]
                weights = np.fabs(np.rad2deg(
                    self.HC_angle[large_HC_idx]))
            # HC angle
            # if variable_name == 'HC_angle':
            #     # TODO: FIX DISTANCE ADJUSTMENT
            #     restricted_HC_start_idx = self.HC_start_idx[
            #         self.distance[self.HC_start_idx] < 69.0]
            #     restricted_HC_angle = self.HC_angle[
            #         self.distance[self.HC_start_idx] < 69.0]
            #     large_HC_idx = np.abs(
            #         restricted_HC_angle) > np.deg2rad(20.)
            #     bearing_angle = self.bearing_angle[
            #         restricted_HC_start_idx[large_HC_idx]]
            #     weights = np.rad2deg(
            #         restricted_HC_angle[large_HC_idx])

            # add data for circular boundary conditions
            bearing_angle = np.hstack(
                [bearing_angle - 2 *
                 np.pi, bearing_angle, bearing_angle + 2 * np.pi])
            weights = np.tile(weights, 3)

            # hist
            edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
            n_samples = np.histogram(bearing_angle, bins=edges_bearing,
                                     normed=False)[0]
            hist = np.histogram(bearing_angle, bins=edges_bearing,
                                normed=False, weights=weights)[0]
            hist = hist / n_samples

            # convolve, filter width = 60 degree
            hist = np.convolve(np.ones(11) / 11., hist, mode='same')
            line_data = np.vstack((line_data, hist))

            # plot
            ax.plot(np.rad2deg(edges_bearing)[:-1], hist, ls='-',
                    lw=self.lw[self.full_condition],
                    color=self.lc[self.full_condition],
                    alpha=3 * self.alpha[self.full_condition],
                    label=self.names[self.full_condition])

        # legend
        leg = ax.legend(
            loc=[
                0,
                1.0],
            ncol=1,
            handlelength=1.5,
            fontsize=font_size)
        frame = leg.get_frame()
        frame.set_linewidth(0.0)
        # save data
        if par['save_data']:
            column_names = list(conditions)
            column_names.insert(0, 'Bearing Angle')
            saved_line_idx = np.abs(line_data[0]) <= 183
            df = pandas.DataFrame(line_data.T[saved_line_idx])
            df.columns = column_names
            df.to_excel(self.excelWriter,
                        sheet_name=str(variable_name) +
                                   'bear_line' + dstrshort,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    str(variable_name) + 'bear_line' + dstrshort])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        str(variable_name) + 'to_bearing_line' + dstrshort)
            plt.close()

    def figure_norm_track_area_explored_boxplot(self, par):

        conditions = [par['condition']]
        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.15, right=0.9, hspace=0., wspace=0.,
                            bottom=0.3, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        area_averages = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        # init
        try:
            area_averages = self.area_averages
            print("self.area_averages doesn't exist?")
        except:
            # for all trials
            for trial in pref_trials:
                area_averages.append(
                    self.norm_individual_area_to_tracklength(
                        par,
                        [trial]))
            self.area_averages = area_averages

        # convert to array
        area_averages = np.array(area_averages)

        # make boxplot
        bp = ax.boxplot(area_averages, positions=[0],
                        widths=0.3, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')
        # yticks = [x / 10. for x in range(-10, 10,1)]
        # figure settings (has to come after boxplot)
        ax.yaxis.set_ticks_position('left')
        plt.setp(ax,  # ylim=(0.0, 1.0),
                 ylabel='Group Area Covered',
                 xlim=(-0.5, len(conditions) - 0.5),
                 xticks=list(range(len(conditions))),
                 # yticks=[0.0, 0.25, 0.5, 0.75, 1.0]
                 )
        # plot zero
        # ax.axhline(0, color='gray')

        ax.set_xticklabels(
            [self.names_short[self.full_condition] for condition in conditions],
            rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        # save data
        if par['save_data']:
            group = np.array(np.tile(self.names_short[self.full_condition],
                                     len(area_averages)))
            df = pandas.DataFrame(np.column_stack((group,
                                                   pref_trials, area_averages)))
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "Normalized Area"]
            df.to_excel(self.excelWriter,
                        sheet_name='Normalized Individual Area',
                        header=["Group", "Trial Name", "Normalized Area"],
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'Normalized Individual Area'])
        # np.savetxt(par['data_dir'] +
        #            '/' + par['experiment_name'] +
        #            '_' + par['group_name'] +
        #            '_' + par['condition'] + '_' +
        #            'PREF.csv', np.column_stack((pref_trials, pref)),
        #            delimiter=',', fmt="%s",
        #            header="TRIAL_NAME, PREF")
        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] +
                        '_' + par['condition'] + '_' +
                        'Normalized_AREA')
            plt.close()

    def figure_group_area_explored_boxplot(self, par):

        conditions = [par['condition']]
        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.15, right=0.9, hspace=0., wspace=0.,
                            bottom=0.3, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        area_ratios = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        # trial_numbers = [self.trial_number[index] for index in sorted(
        #     np.unique(self.trial_number, return_index=True)[1])
        # ]
        # for all trials
        for trial in pref_trials:
            area_ratios.append(self.group_area_trial_tracks_on_dish(par,
                                                                    [trial]))

        # convert to array
        area_ratios = np.array(area_ratios)

        # make boxplot
        bp = ax.boxplot(area_ratios, positions=[0],
                        widths=0.3, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')
        # yticks = [x / 10. for x in range(-10, 10,1)]
        # figure settings (has to come after boxplot)
        ax.yaxis.set_ticks_position('left')
        plt.setp(ax, ylim=(0.0, 1.0),
                 ylabel='Group Area Covered',
                 xlim=(-0.5, len(conditions) - 0.5),
                 xticks=list(range(len(conditions))),
                 yticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        # plot zero
        # ax.axhline(0, color='gray')

        ax.set_xticklabels(
            [self.names_short[self.full_condition] for condition in conditions],
            rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        # save data
        if par['save_data']:
            group = np.array(np.tile(self.names_short[self.full_condition],
                                     len(area_ratios)))
            df = pandas.DataFrame(np.column_stack((group,
                                                   pref_trials, area_ratios)))
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "Group Area Ratio"]
            df.to_excel(self.excelWriter,
                        sheet_name='Group Area Explored',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'Group Area Explored'])
        # np.savetxt(par['data_dir'] +
        #            '/' + par['experiment_name'] +
        #            '_' + par['group_name'] +
        #            '_' + par['condition'] + '_' +
        #            'PREF.csv', np.column_stack((pref_trials, pref)),
        #            delimiter=',', fmt="%s",
        #            header="TRIAL_NAME, PREF")
        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] +
                        '_' + par['condition'] + '_' +
                        'Group AREA')
            plt.close()

    def figure_boxplot_PREF(self, par, figure_name='boxplot_PREF'):

        conditions = [par['condition']]
        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.15, right=0.9, hspace=0., wspace=0.,
                            bottom=0.3, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        pref = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        # trial_dict = dict(zip(trial_numbers, pref_trials))
        # for all trials
        for trial_number in trial_numbers:
            # init
            time_up = 0.
            time_down = 0.

            # select trial and not nan idx
            trial_idx = self.trial_number == trial_number
            not_nan_idx = ~np.isnan(
                self.spine4[:, 0])

            # spine4_y
            spine4_y = self.spine4[
                not_nan_idx *
                trial_idx,
                1]

            # time up and down
            time_down = np.sum(spine4_y < 0.) * float(par['dt'])
            time_up = np.sum(spine4_y > 0.) * float(par['dt'])

            pref.append(
                (time_up - time_down) / (time_down + time_up).astype(float))

        # convert to array
        pref = np.array(pref)

        # make boxplot
        # PLOTLY
        # ===================
        # data = [pg.Box(
        #     y=pref,
        #     boxpoints='all',
        #     jitter=0.3,
        #     pointpos=-1.8,
        #     name='Mean PREF'
        # )
        # ]
        # plotly.offline.plot(data, filename="/tmp/foo.html",
        #                     show_link=False,
        #                     image='svg',
        #                     image_height=2048,
        #                     image_width=2048
        #                     )
        bp = ax.boxplot(pref, positions=[0],
                        widths=0.3, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')
        # yticks = [x / 10. for x in range(-10, 10,1)]
        # figure settings (has to come after boxplot)
        ax.yaxis.set_ticks_position('left')
        plt.setp(ax, ylim=(-1.0, 1.0),
                 ylabel='Mean PREF', xlim=(-0.5, len(conditions) - 0.5),
                 xticks=list(range(len(conditions))),
                 yticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
        # plot zero
        ax.axhline(0, color='gray')

        ax.set_xticklabels(
            [self.names_short[self.full_condition] for condition in conditions],
            rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        # save data
        if par['save_data']:
            group = np.array(np.tile(self.names_short[self.full_condition],
                                     len(pref)))
            df = pandas.DataFrame(np.column_stack((group,
                                                   pref_trials,
                                                   pref.astype(float))))
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "PREF"]
            df.to_excel(self.excelWriter,
                        sheet_name='PREF',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'PREF'])

        # np.savetxt(par['data_dir'] +
        #            '/' + par['experiment_name'] +
        #            '_' + par['group_name'] +
        #            '_' + par['condition'] + '_' +
        #            'PREF.csv', np.column_stack((pref_trials, pref)),
        #            delimiter=',', fmt="%s",
        #            header="TRIAL_NAME, PREF")
        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] +
                        '_' + par['condition'] + '_' +
                        'PREF')
            plt.close()

    def figure_boxplot_variable_depending_on_bearing(
            self,
            par,
            variable_name,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        # this function takes very long to compute, because of ...== trial
        # dstr = ""
        dstrshort = ""

        # figure settings
        ylabel = {
            'INS_interval': 'Inter-step-interval (s)',
            'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate (1/s)',
            'HC_angle': 'HC angle (' + degree_sign + ')',
            'Abs_HC_angle': 'ABS HC angle (' + degree_sign + ')',
            'run_speed': 'Run Speed(mm/s)'
        }

        column_names = {
            'INS_interval': ['toward', 'away'],
            'INS_turn': ['left', 'right'],
            'INS_distance': ['toward', 'away'],
            'HC_rate': ['toward', 'away'],
            'run_speed': ['toward', 'away'],
            'HC_angle': ['left', 'right'],
            'Abs_HC_angle': ['turn toward', 'turn away'],
        }

        # ylim = {
        #     'INS_interval': (0.8, 0.9),
        #     'INS_turn': (-1.4, 1.4),
        #     'INS_distance': (0.7, 0.8),
        #     'HC_rate': (0.3, 0.7),
        #     'HC_angle': (-20, 20),
        #     }

        # yticks = {
        #     'INS_interval': [0.8, 0.85, 0.9],
        #     'INS_turn': [-1.4, 0, 1.4],
        #     'INS_distance': [0.7, 0.75, 0.8],
        #     'HC_rate': [0.3, 0.5, 0.7],
        #     'HC_angle': [-20, -10, 0, 10, 20],
        #     }

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot_black = []
        boxplot_white = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition
        # for all trials
        for trial_number in trial_numbers:

            # mean_INS_distance,
            # mean_INS_interval, step_turning_angle
            if variable_name in ['INS_distance',
                                 'INS_interval',
                                 'INS_turn']:

                idx_trial = (
                        self.trial_number[
                            self.step_idx[
                                self.next_event_is_step]] == trial_number)

                bearing_angle = self.bearing_angle[
                    self.step_idx[
                        self.next_event_is_step]]
                bearing_angle = bearing_angle[idx_trial]

                weights = getattr(self, variable_name)[
                    self.next_event_is_step]
                weights = weights[idx_trial]

                if variable_name == 'INS_turn':
                    weights = np.rad2deg(weights)

                # delete very unrealistic outliers
                if variable_name == 'INS_distance':
                    idx_ok = weights < 1.5  # must be smaller than 1.5 mm
                    bearing_angle = bearing_angle[idx_ok]
                    weights = weights[idx_ok]

                if variable_name == 'INS_interval':
                    idx_ok = weights < 10  # must be smaller than 10 seconds
                    bearing_angle = bearing_angle[idx_ok]
                    weights = weights[idx_ok]

            # HC rate
            if variable_name == 'HC_rate':
                idx_not_nan = ~np.isnan(self.HC_initiation)
                idx_trial = self.trial_number == trial_number
                bearing_angle = self.bearing_angle[
                    idx_not_nan * idx_trial]
                weights = self.HC_initiation[
                              idx_not_nan * idx_trial] / float(par['dt'])

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(self.midpoint_speed)
                idx_non_hc = self.HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                idx_non_hc = idx_non_hc * idx_not_nan
                idx_trial = self.trial_number == trial_number
                bearing_angle = self.bearing_angle[
                    idx_non_hc * idx_trial]
                # weights = self.dict[self.full_condition].centroid_speed[
                #     idx_non_hc * idx_trial]
                weights = self.midpoint_speed[
                    idx_non_hc * idx_trial]

            # HC angle
            if variable_name in ['HC_angle', 'Abs_HC_angle']:
                if subthreshold:
                    # dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    # dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(self.HC_angle, large_HC,
                                         subthreshold)

                bearing_angle = self.bearing_angle[
                    self.HC_start_idx[large_HC_idx]]
                weights = np.rad2deg(
                    self.HC_angle[large_HC_idx])

                idx_trial = (
                        self.trial_number
                        [self.HC_start_idx[large_HC_idx]] ==
                        trial_number)

                bearing_angle = bearing_angle[idx_trial]
                weights = weights[idx_trial]

            # apend boxplotdata
            if variable_name in ['INS_distance',
                                 'INS_interval']:
                idx_black = np.abs(bearing_angle) < par['to_range']
                idx_white = np.abs(bearing_angle) > par['away_range']

                boxplot_black.append(np.mean(weights[idx_black]))
                boxplot_white.append(np.mean(weights[idx_white]))
                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][0],
                                     np.mean(weights[idx_black])])
                boxplot_data.append([full_condition,
                                     trial_dict[trial_number],
                                     column_names[variable_name][1],
                                     np.mean(weights[idx_white])])

            if variable_name in ['HC_rate', 'run_speed']:
                idx_black = np.abs(bearing_angle) < par['to_range']
                idx_white = np.abs(bearing_angle) > par['away_range']

                boxplot_black.append(
                    np.sum(weights[idx_black]) / len(weights[idx_black]))
                boxplot_white.append(
                    np.sum(weights[idx_white]) / len(weights[idx_white]))
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][0],
                    np.sum(weights[idx_black]) / len(weights[idx_black])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][1],
                    np.sum(weights[idx_white]) / len(weights[idx_white])
                ])

            if variable_name in ['Abs_HC_angle']:
                # Towards
                idx_black = bearing_angle * weights < 0.
                # Away
                idx_white = bearing_angle * weights > 0.

                boxplot_black.append(np.mean(np.fabs(weights[idx_black])))
                boxplot_white.append(np.mean(np.fabs(weights[idx_white])))
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][0],
                    np.mean(np.fabs(weights[idx_black]))
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][1],
                    np.mean(np.fabs(weights[idx_white]))
                ])

            if variable_name in ['INS_turn', 'HC_angle']:
                idx_black = bearing_angle < 0.
                idx_white = bearing_angle > 0.

                boxplot_black.append(np.mean(weights[idx_black]))
                boxplot_white.append(np.mean(weights[idx_white]))
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][0],
                    np.mean(weights[idx_black])
                ])
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + column_names[variable_name][1],
                    np.mean(weights[idx_white])
                ])

        # make black boxplot
        bp = ax.boxplot(boxplot_black, positions=[0],
                        widths=0.15, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # make white boxplot
        bp = ax.boxplot(boxplot_white, positions=[1],
                        widths=0.15, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel[variable_name],
            xlim=(-0.5, 1.5),
            xticks=list(range(2)))
        ax.set_xticklabels(
            column_names[variable_name],
            rotation=45, ha='right', size=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        ax.annotate(par['condition'], xy=(0.75, 1.0),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          '/'.join(column_names[variable_name]),
                          str(variable_name)]
            df.to_excel(self.excelWriter,
                        sheet_name=str(variable_name) +
                                   'bear_box' + dstrshort,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[str(variable_name) +
                                        'bear_box' + dstrshort]
            )
        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        str(variable_name) + 'to_bearing_box' + dstrshort)
            plt.close()

    # def figure_boxplot_proportion_of_HC(
    #         self, par):
    #     ylabel = 'proportion of HCs'
    #     column_names = ['toward', 'away']

    #     # figure
    #     fig = plt.figure(
    #         figsize=(
    #             par['fig_width'],
    #             par['fig_width']))
    #     fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
    #                         bottom=0.25, top=0.9)

    #     # init
    #     boxplot_data = []
    #     boxplot_black = []
    #     boxplot_white = []
    #     pref_trials = np.unique(self.trial)
    #     trial_numbers = self.trial_number
    #     full_condition = self.full_condition
    #     # for all trials
    #     for trial_number in np.unique(trial_numbers):
    #         large_HC_idx = np.abs(
    #             self.HC_angle) > np.deg2rad(20.)
    #         print self.HC_angle
    #         bearing_angle = self.bearing_angle[
    #             self.HC_start_idx[large_HC_idx]]
    #         weights = np.rad2deg(
    #             self.HC_angle[large_HC_idx])
    #         bearing_angle = bearing_angle[idx_trial]
    #         weights = weights[idx_trial]
    #     multiplied = bearing_angle * weights

    def figure_proportion_of_HCs_boxplot(
            self,
            par,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        ylabel = "Proportion of HCs"
        column_names = ['To odor', 'Away from odor']
        # column_names_short = ['towards', 'away']

        if subthreshold:
            # dstr = "(max angle " + str(large_HC) + ")"
            dstrshort = "_maxA" + str(int(large_HC))
        else:
            # dstr = "(min angle " + str(large_HC) + ")"
            dstrshort = "_minA" + str(int(large_HC))

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot_black = []
        boxplot_white = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition
        # for all trials
        for trial_number in trial_numbers:
            large_HC_idx = angleComp(self.HC_angle, large_HC,
                                     subthreshold)

            bearing_angle = self.bearing_angle[
                self.HC_start_idx[large_HC_idx]]
            idx_trial = (
                    self.trial_number
                    [self.HC_start_idx[large_HC_idx]] ==
                    trial_number)
            bearing_angle = bearing_angle[idx_trial]

            weights = np.rad2deg(
                self.HC_angle[large_HC_idx])

            weights = weights[idx_trial]
            idx_towards = np.sum(bearing_angle * weights < 0.)
            idx_away = np.sum(bearing_angle * weights > 0.)
            if ((idx_away + idx_towards) == 0):
                continue
            proportion_towards = (
                    float(idx_towards) / float(idx_away + idx_towards))
            proportion_away = float(idx_away) / float(idx_away + idx_towards)

            boxplot_black.append(proportion_towards)
            boxplot_white.append(proportion_away)
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                proportion_towards,
                proportion_away
            ])

        # make black boxplot
        bp = ax.boxplot(boxplot_black, positions=[0],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # make white boxplot
        bp = ax.boxplot(boxplot_white, positions=[1],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel,
            xlim=(-0.5, 1.5),
            ylim=(0.0, 1.0),
            xticks=[0, 1.0],
            yticks=np.arange(0.0, 1.1, 0.25))
        ax.set_xticklabels(
            column_names,
            rotation=45, ha='right', size=font_size)
        # plot zero
        # ax.set_yticklabels(
        #    np.arange(0.0,1.0,0.1),
        #    rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.axhline(0.5, color='lightgray', zorder=-1)

        ax.annotate(par['condition'], xy=(0.75, 1.0),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          column_names[0],
                          column_names[1]]
            df.to_excel(self.excelWriter,
                        sheet_name='proportions_to_box' + dstrshort,
                        index=False)

            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'proportions_to_box' + dstrshort])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        'proportions_to_boxplot' + dstrshort)
            plt.close()

    def figure_proportion_of_HCs_boxplot_distance_split(
            self,
            par,
            distance,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        ylabel = "Proportion of HCs to odor"
        column_names = ['near', 'far']

        if subthreshold:
            # dstr = "(max angle " + str(large_HC) + ")"
            dstrshort = "_maxA" + str(int(large_HC))
        else:
            # dstr = "(min angle " + str(large_HC) + ")"
            dstrshort = "_minA" + str(int(large_HC))

        # column_names_short = ['towards', 'away']

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot_black = []
        boxplot_white = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition

        near_HC_start_idx = self.HC_start_idx[
            self.distance[self.HC_start_idx] < distance]
        far_HC_start_idx = self.HC_start_idx[
            self.distance[self.HC_start_idx] >= distance]

        near_HC_angle = self.HC_angle[
            self.distance[self.HC_start_idx] < distance]
        far_HC_angle = self.HC_angle[
            self.distance[self.HC_start_idx] >= distance]

        near_large_HC_idx = angleComp(near_HC_angle, large_HC,
                                      subthreshold)
        far_large_HC_idx = angleComp(far_HC_angle, large_HC,
                                     subthreshold)

        # for all trials
        for trial_number in trial_numbers:

            near_idx_trial = (
                    self.trial_number
                    [near_HC_start_idx[near_large_HC_idx]] ==
                    trial_number)
            far_idx_trial = (
                    self.trial_number
                    [far_HC_start_idx[far_large_HC_idx]] ==
                    trial_number)

            near_bearing_angle = self.bearing_angle[
                near_HC_start_idx[near_large_HC_idx]]
            far_bearing_angle = self.bearing_angle[
                far_HC_start_idx[far_large_HC_idx]]

            near_weights = np.rad2deg(
                near_HC_angle[near_large_HC_idx])
            far_weights = np.rad2deg(
                far_HC_angle[far_large_HC_idx])

            near_bearing_angle = near_bearing_angle[near_idx_trial]
            far_bearing_angle = far_bearing_angle[far_idx_trial]

            near_weights = near_weights[near_idx_trial]
            far_weights = far_weights[far_idx_trial]

            # DEBUG
            # print '=================PROPORTIONS==============='
            # print "trial number: " + str(trial_number)
            # print "near_HC_angle: " + str(near_HC_angle.shape)
            # print "far_HC_angle: " + str(far_HC_angle.shape)
            # print "near_large_HC_idx: " + str(near_large_HC_idx.shape)
            # print "far_large_HC_idx: " + str(far_large_HC_idx.shape)
            # print "near_idx_trial: " + str(near_idx_trial.shape)
            # print "far_idx_trial: " + str(far_idx_trial.shape)
            # print "near_bearing_angle: " + str(near_bearing_angle.shape)
            # print "far_bearing_angle: " + str(far_bearing_angle.shape)
            # print "near_weights: " + str(near_weights.shape)
            # print "far_weights: " + str(far_weights.shape)

            idx_towards_near = np.sum(near_bearing_angle * near_weights < 0.)
            idx_towards_far = np.sum(far_bearing_angle * far_weights < 0.)
            idx_away_near = np.sum(near_bearing_angle * near_weights > 0.)
            idx_away_far = np.sum(far_bearing_angle * far_weights > 0.)
            if ((idx_away_near + idx_towards_near) == 0):
                proportion_towards_near = 0
            else:
                proportion_towards_near = float(idx_towards_near) / float(
                    idx_away_near + idx_towards_near)

            if ((idx_away_far + idx_towards_far) == 0):
                proportion_towards_far = 0
            else:
                proportion_towards_far = float(idx_towards_far) / float(
                    idx_away_far + idx_towards_far)

            boxplot_black.append(proportion_towards_near)
            boxplot_white.append(proportion_towards_far)
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                full_condition + ' near',
                proportion_towards_near,
            ])
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                full_condition + ' far',
                proportion_towards_far,
            ])

        # make black boxplot
        bp = ax.boxplot(boxplot_black, positions=[0],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # make white boxplot
        bp = ax.boxplot(boxplot_white, positions=[1],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel,
            xlim=(-0.5, 1.5),
            ylim=(0.0, 1.0),
            xticks=[0, 1.0],
            yticks=np.arange(0.0, 1.1, 0.25))
        ax.set_xticklabels(
            column_names,
            rotation=45, ha='right', size=font_size)
        # plot zero
        # ax.set_yticklabels(
        #    np.arange(0.0,1.0,0.1),
        #    rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.axhline(0.5, color='lightgray', zorder=-1)

        ax.annotate(par['condition'] + ' (d = ' + str(distance) + ')',
                    xy=(0.75, 1.0),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          '/'.join(column_names),
                          'value']
            df.to_excel(self.excelWriter,
                        sheet_name='proportions_to_box_d' + dstrshort,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'proportions_to_box_d' + dstrshort])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        'proportions_to_boxplot_d' + dstrshort)
            plt.close()

    def figure_HC_reorientation_boxplot_distance_split(
            self,
            par,
            distance,
            reorientation=True,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']
        if reorientation:
            dstr = 'reorientation'
            ylabel = "HC reorientation"
        else:
            dstr = 'accuracy'
            ylabel = "HC accuracy"

        if subthreshold:
            dstr = dstr + " (max angle " + str(large_HC) + ")"
            dstrshort = "_maxA" + str(int(large_HC))
        else:
            dstr = dstr + "(min angle " + str(large_HC) + ")"
            dstrshort = "_minA" + str(int(large_HC))

        # column_names = ['Reorientation/near', 'reorientation/near',
        #                 'Reorientation/far', 'reorientation/far']
        column_names = [dstr + '/near',
                        dstr + '/far']
        # column_names_short = ['towards', 'away']

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot_black = []
        boxplot_white = []
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition

        near_HC_start_idx = self.HC_start_idx[
            self.distance[self.HC_start_idx] < distance]
        near_HC_end_idx = self.HC_end_idx[
            self.distance[self.HC_start_idx] < distance]
        far_HC_start_idx = self.HC_start_idx[
            self.distance[self.HC_start_idx] >= distance]
        far_HC_end_idx = self.HC_end_idx[
            self.distance[self.HC_start_idx] >= distance]

        near_HC_angle = self.HC_angle[
            self.distance[self.HC_start_idx] < distance]
        far_HC_angle = self.HC_angle[
            self.distance[self.HC_start_idx] >= distance]

        # angle = par['large_HC']
        # angle = 0
        near_large_HC_bidx = angleComp(near_HC_angle, large_HC,
                                       subthreshold)
        far_large_HC_bidx = angleComp(far_HC_angle, large_HC,
                                      subthreshold)

        # for all trials
        for trial_number in trial_numbers:

            near_idx_trial = (
                    self.trial_number
                    [near_HC_start_idx[near_large_HC_bidx]] ==
                    trial_number)
            near_idx_trial_at_end = (
                    self.trial_number
                    [near_HC_end_idx[near_large_HC_bidx]] ==
                    trial_number)
            far_idx_trial = (
                    self.trial_number
                    [far_HC_start_idx[far_large_HC_bidx]] ==
                    trial_number)
            # print far_HC_start_idx[0:10]
            # print far_HC_start_idx.shape
            # print far_HC_end_idx[0:10]
            # print far_HC_end_idx.shape
            # print far_large_HC_bidx[0:10]
            # print far_large_HC_bidx.shape
            # print self.trial_number[0:10]
            # print self.trial_number.shape
            # print near_idx_trial[0:10]
            # print near_idx_trial.shape

            far_idx_trial_at_end = (
                    self.trial_number
                    [far_HC_end_idx[far_large_HC_bidx]] ==
                    trial_number)

            near_bearing_angle_at_start = self.bearing_angle[
                near_HC_start_idx[near_large_HC_bidx]][
                near_idx_trial]
            # near_heading_angle_at_start = self.heading_angle[
            #     near_HC_start_idx[near_large_HC_idx]]
            near_heading_angle_at_end = self.heading_angle[
                near_HC_end_idx[near_large_HC_bidx]][
                near_idx_trial_at_end]

            far_bearing_angle_at_start = self.bearing_angle[
                far_HC_start_idx[far_large_HC_bidx]][
                far_idx_trial]
            # far_heading_angle_at_start = self.heading_angle[
            #     far_HC_start_idx[far_large_HC_idx]]
            far_heading_angle_at_end = self.heading_angle[
                far_HC_end_idx[far_large_HC_bidx]][
                far_idx_trial_at_end]

            # near_weights = np.rad2deg(
            #     np.abs(near_bearing_angle_at_start) -
            #     np.abs(near_heading_angle_at_end))
            # far_weights = np.rad2deg(
            #     np.abs(far_bearing_angle_at_start) -
            #     np.abs(far_heading_angle_at_end))
            if reorientation:
                near_weights = np.rad2deg(
                    np.abs(near_bearing_angle_at_start) -
                    np.abs(near_heading_angle_at_end))
                far_weights = np.rad2deg(
                    np.abs(far_bearing_angle_at_start) -
                    np.abs(far_heading_angle_at_end))
            else:
                near_weights = np.rad2deg(
                    np.abs(near_heading_angle_at_end))
                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_end))
            # near_bearing_angle = near_bearing_angle[near_idx_trial]
            # far_bearing_angle = far_bearing_angle[far_idx_trial]

            # near_weights = near_weights[near_idx_trial]
            # far_weights = far_weights[far_idx_trial]

            # DEBUG
            # print '=================PROPORTIONS==============='
            # print "trial number: " + str(trial_number)
            # print "near_HC_angle: " + str(near_HC_angle.shape)
            # print "far_HC_angle: " + str(far_HC_angle.shape)
            # print "near_large_HC_idx: " + str(near_large_HC_idx.shape)
            # print "far_large_HC_idx: " + str(far_large_HC_idx.shape)
            # print "near_idx_trial: " + str(near_idx_trial.shape)
            # print "far_idx_trial: " + str(far_idx_trial.shape)
            # print "near_bearing_angle: " + str(near_bearing_angle.shape)
            # print "far_bearing_angle: " + str(far_bearing_angle.shape)
            # print "near_weights: " + str(near_weights.shape)
            # print "far_weights: " + str(far_weights.shape)

            weight_average_near = np.sum(near_weights) / len(near_weights)
            weight_average_far = np.sum(far_weights) / len(far_weights)
            boxplot_black.append(weight_average_near)
            boxplot_white.append(weight_average_far)
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                full_condition + ' near',
                weight_average_near,
            ])
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                full_condition + ' far',
                weight_average_far,
            ])

        # make black boxplot
        bp = ax.boxplot(boxplot_black, positions=[0],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # make white boxplot
        bp = ax.boxplot(boxplot_white, positions=[1],
                        widths=0.25, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='gray')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='gray')
        plt.setp(bp['whiskers'], color='gray', ls='-')
        plt.setp(bp['fliers'], color='gray', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel,
            xlim=(-0.5, 1.5),
            # ylim=(-20, 30),
            xticks=[0, 1.0],
            # yticks=np.arange(-20, 31, 5)
        )
        ax.set_xticklabels(
            column_names,
            rotation=45, ha='right', size=font_size)
        # plot zero
        # ax.set_yticklabels(
        #    np.arange(0.0,1.0,0.1),
        #    rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.axhline(0.0, color='lightgray', zorder=-1)

        ax.annotate(par['condition'] + ' (d = ' + str(distance) + ')',
                    xy=(0.75, 1.0),
                    xycoords='axes fraction', size=font_size,
                    horizontalalignment='right',
                    verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          '/'.join(column_names),
                          'value']
            df.to_excel(self.excelWriter,
                        sheet_name='HC_' + dstrshort + '_box_d',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'HC_' + dstrshort + '_box_d'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        'HC_' + dstrshort + '_box_d')
            plt.close()

    def figure_run_speed_boxplot(
            self,
            par
    ):

        ylabel = 'Run Speed (mm/s)'

        # figure
        fig = plt.figure(
            figsize=(
                par['fig_width'],
                par['fig_width']))
        fig.subplots_adjust(left=0.2, right=0.9, hspace=0., wspace=0.,
                            bottom=0.25, top=0.9)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # init
        boxplot_data = []
        boxplot = []
        full_condition = self.full_condition
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [self.trial_number[index] for index in sorted(
            np.unique(self.trial_number, return_index=True)[1])
                         ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        # for all trials
        for trial_number in trial_numbers:
            # Run Speed
            idx_not_nan = ~np.isnan(self.midpoint_speed)
            idx_non_hc = self.HC == 0
            # Leave some distance before and after HC
            idx_non_hc = np.invert(np.convolve(
                np.invert(idx_non_hc),
                (par['gap'] * 2 + 1) * [1], mode='same') > 0)

            idx_non_hc = idx_non_hc * idx_not_nan
            # PAD the HC for run speed calculation (shortly before and after
            #  HC speed should be slower). 16 frames before and after
            # idx_non_hc = (np.convolve(
            #    (idx_non_hc == 0), np.ones(16*2+1), 'same') == 0)
            idx_trial = self.trial_number == trial_number
            weights = self.midpoint_speed[
                idx_non_hc * idx_trial]

            boxplot.append(
                np.sum(weights) / len(weights))
            boxplot_data.append([
                full_condition,
                trial_dict[trial_number],
                np.sum(weights) / len(weights)
            ])

        # make black boxplot
        bp = ax.boxplot(boxplot, positions=[0],
                        widths=0.15, whis=1.6, sym='')
        plt.setp(bp['boxes'], color='k')
        plt.setp(bp['medians'], color='r')
        plt.setp(bp['caps'], color='k')
        plt.setp(bp['whiskers'], color='k', ls='-')
        plt.setp(bp['fliers'], color='k', marker='+')

        # figure settings (has to come after boxplot)
        plt.setp(
            ax, ylabel=ylabel,
            xlim=(-0.5, 0.5),
            xticks=list(range(2)))
        ax.set_xticklabels(
            [par['experiment_name'] + par['condition']],
            rotation=45, ha='right', size=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(3))

        # ax.annotate(par['condition'], xy=(0.75, 1.0),
        #             xycoords='axes fraction', size=font_size,
        #             horizontalalignment='right',
        #             verticalalignment='bottom', rotation=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          'Run Speed']
            df.to_excel(self.excelWriter,
                        sheet_name='Run Speed',
                        index=False)

            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'Run Speed'])
        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        '_runspeed_boxplot')
            plt.close()
