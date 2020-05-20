
# ==============================================================================
# Individual class multiple groups
# ==============================================================================
import pandas
import os
import numpy
from Tracks import Tracks
import copy
# from Plotlyplots import *
import numpy as np
from MiscFunctions import MiscFunctions as mf
import operator

from scipy.stats import binned_statistic
import mysql.connector

class Individuals:
    '''class for analysis of individuals from multiple groups'''

    def __init__(self,
                 par,
                 experiment,
                 grouped_conditions,
                 which_trials='all',
                 which_tracks='all'):

        mydb = mysql.connector.connect(
            host="localhost",
            user="mthane",
            passwd="appy",
            database="test_db"
            )
        mycursor = mydb.cursor()
        mycursor.execute("drop table individuals")

        mycursor.execute("CREATE TABLE IF NOT EXISTS individuals"+" (track_name VARCHAR(255),  trial_name VARCHAR(255))")


        self.experiment = experiment
        self.dict = {}
        groups_name = '--'.join(
            str(p.replace('/', '_')) for p in par['groups'])
        if par['save_data']:
            self.excelWriter = pandas.ExcelWriter(
                "individuals"+
                '.xlsx', engine='xlsxwriter')

        self.groups_name = groups_name
        self.plotly_included = False
        self.plotly_filename = par['figure_dir'] + "/"
        self.plotly_columns = []
        self.excel_titles = {}
        self.total_tracks = 0
        if par['save_data']:
            df = pandas.DataFrame()
            df.to_excel(self.excelWriter,
                        sheet_name='Table of Contents',
                        index=False)
        group_condition_array = [x.split('_-_') for x in grouped_conditions]
        tmpdir = (par['parent_dir'] + '/' +
                  par['experiment_name'] + '/' +
                  par['tracked_on'] + '/' + 'tmp')
        for group_condition in group_condition_array:
            print("====" + str(group_condition), flush=True)
            group = group_condition[0]
            condition = group_condition[1]
            condition_pkl = (
                    tmpdir + '/' +
                    experiment +
                    '_' +
                    group +
                    '_-_' +
                    condition +
                    par['path_suffix'])
            if os.path.exists(condition_pkl):
                self.dict[group + '_-_' + condition] = mf.load_pkl(
                    tmpdir + '/' +
                    experiment +
                    '_' +
                    group +
                    '_-_' +
                    condition +
                    par['path_suffix'])
            else:
                par_group = par
                par_group['group_name'] = group
                par_group['condition'] = condition
                self.dict[group + '_-_' + condition] = Tracks(
                    par_group, 'all', 'all')

        conditions = grouped_conditions
        self.names = conditions
        self.names_short = self.names

        colors = ['r', 'g', 'c', 'orange', 'blue', 'black', 'purple',
                  'pink', 'magenta', 'goldenrod', 'saddlebrown', 'grey']

        self.lc = dict(list(zip(conditions, colors)))

        linewidths = [1.0] * 12
        self.lw = dict(list(zip(conditions, linewidths)))
        alphas = [1.0] * 12
        self.alpha = dict(list(zip(conditions, alphas)))
        track_idx = 0
        for cond in conditions:

            self.dict[cond].total_track_count = 0
            self.dict[cond].accepted_tracks = []
            self.dict[cond].not_accepted_tracks = []
            self.dict[cond].accepted_track_count = 0
            track_count = 0
            # for all trials
            pref_trials = [
                self.dict[cond].trial[index] for index in sorted(
                    np.unique(self.dict[cond].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[cond].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[cond].trial_number,
                        return_index=True)[0])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))

            avg_size = 0
            for trial_number in trial_numbers:
                trial_bidx = self.dict[cond].trial_number == trial_number
                trial_tracks = np.unique(
                    self.dict[cond].track_number[trial_bidx])
                # Dish radius:
                R = par['radius_dish']
                # Distance odor-center of dish
                if len(np.array(self.dict[cond].odor_A).shape)==1:
                    distance = self.dict[cond].odor_A[0]
                else:
                    distance = self.dict[cond].odor_A[trial_number][0]
                # Radius of area near odor covering third of the petri-dish area
                trial_radius = mf.find_r(R * R * np.pi / 2.0, R, distance)
                for trial_track in trial_tracks:
                    # print "Checking: Track number " +str(track_count)
                    # + " -> "  + str(trial_number)
                    # + ".track(" + str(trial_track) + ")"

                    track_bidx = self.dict[cond].track_number == trial_track
                    track_bidx = track_bidx * trial_bidx
                    track_idx = np.where(track_bidx == True)
                    track_step_bidx = np.in1d(
                        self.dict[cond].step_idx,
                        track_idx,
                        assume_unique=True
                    )

                    track_HC_bidx = np.in1d(
                        self.dict[cond].HC_start_idx,
                        track_idx,
                        assume_unique=True
                    )

                    tlen = (
                            self.dict[cond].time[track_bidx][-1] -
                            self.dict[cond].time[track_bidx][0])
                    if np.isnan(tlen):
                        avg_size = avg_size + np.nansum(track_bidx)
                    else:
                        avg_size = avg_size + (tlen * par['fps'])

                    if self.check_track(par, cond, track_count, track_bidx):
                        self.dict[cond].accepted_tracks.append(
                            {"trial": trial_dict[trial_number],
                             "trial_number": trial_number,
                             "track_number": trial_track,
                             "track_id": track_count,
                             "track_bidx": track_bidx,
                             "track_HC_bidx": track_HC_bidx,
                             "track_step_bidx": track_step_bidx,
                             "trial_radius": trial_radius,
                             "PREF": -2
                             })

                        mydb = mysql.connector.connect(
                                 host="localhost",
                                 user="mthane",
                                 passwd="appy",
                                 database="test_db"
                              )

    
                        mycursor = mydb.cursor()

                        sql = "INSERT IGNORE INTO individuals (track_name, trial_name) VALUES (%s, %s)"
                        val = (str(trial_track), str(trial_dict[trial_number]))
                        print(sql)
                        print(val)
                        mycursor.execute(sql, val)
                        mydb.commit()
                        
                        print(mycursor.rowcount, "record inserted.")
                        mycursor.close()
                    else:
                        self.dict[cond].not_accepted_tracks.append(
                            {"trial": trial_dict[trial_number],
                             "trial_number": trial_number,
                             "track_number": trial_track,
                             "track_id": track_count,
                             "track_bidx": track_bidx,
                             "track_HC_bidx": track_HC_bidx,
                             "track_step_bidx": track_step_bidx,
                             "trial_radius": trial_radius,
                             "PREF": -2
                             })

                track_count = track_count + 1

            self.dict[cond].total_track_count = track_count

            print("Accepted_tracks for " + str(cond) + " :", flush=True)
            #print((str(self.dict[cond].accepted_track_count) +
            #      " / " + str(track_count)))

            ##print("Average Track length: " + str(avg_size / track_count), flush=True)
            mycursor.close()
            mydb.close()
    def check_track(self, par, condition, track_id, track_bidx):
        cond = self.dict[condition]
        # Minimum individual requirement more than half the duration of the
        # time range we want data on
        if (cond.time[track_bidx].shape[0] / par['fps'] < (
                par['end_time'] - par['start_time']) / 2):
            # print ("        - We cannot guarantee identity for track " +
            #        str(track_id))
            # print "Duration: " + str(cond.time[track_bidx].shape)
            cond.accepted_track_count = cond.accepted_track_count + 1

            return True

        # else:
        # print ("        - We CAN guarantee identity for track " +
        #        str(track_id))
        # print "Duration: " + str(cond.time[track_bidx].shape)

        # print "Minimum duration: " + str(par['individual_duration_min'])
        # attrs = vars(track)
        # print ', '.join("%s\n" % item[0] for item in attrs.items())
        # print track.duration
        if track_id is None:
            return False
        # if cond.duration[track_id] >= float(par['individual_duration_min']):
        #    cond.accepted_track_count = cond.accepted_track_count + 1
        #    return True
        else:
            # print ("        - Track: " + str(track.track_number) +
            #        " with duration: " +
            #        str(track.duration) + " too short. Skipping")
            return False


    def write_html(self, filename):
        renderPlotlyHTML(
            self.plotly_filename + filename + ".html",
            self.groups_name,
            self.plotly_columns
        )
        self.plotly_columns = []
        self.plotly_included = False

    def makeExcelTOC(self):
        df = pandas.DataFrame()
        sorted_titles = sorted(
            list(self.excel_titles.items()), key=operator.itemgetter(1))

        for i in sorted_titles:
            df = df.append(pandas.DataFrame(
                [{'Table of Contents': '=HYPERLINK("#' + i[0] + '!A1", "' +
                                       i[1] + '")'}]),
                ignore_index=True)

            df.to_excel(self.excelWriter,
                        sheet_name='Table of Contents',
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets['Table of Contents']
            )

    def figure_variable_depending_on_bearing(self, par, conditions,
                                             variable_name,
                                             subthreshold=False,
                                             large_HC=-1,
                                             ):

        if (large_HC == -1):
            large_HC = par['large_HC']

        dstr = ""
        dstrshort = ""
        if variable_name in ["Abs_HC_angle_turn_TA"]:
            return
        if variable_name in ["Abs_HC_angle_head_TA"]:
            variable_name = "Abs_HC_angle"
        # figure settings
        # ylabel = {
        #     'INS_interval': 'Inter-step-interval (s)',
        #     'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
        #     'INS_distance': 'Inter-step-distance (mm)',
        #     'HC_rate': 'HC rate (1/s)',
        #     'HC_angle': 'HC angle (' + degree_sign + ')',
        #     'Abs_HC_angle': 'Absolute HC angle (' +
        #     degree_sign + ')',
        #     'run_speed': 'Speed (mm/s)'
        # }

        # n_cond = len(conditions)
        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 50)
        line_data = np.rad2deg(edges_bearing)[:-1]
        # for all conditions
        # print conditions
        for condition in conditions:
            cur_cond = self.dict[condition]
            # mean_INS_distance, mean_INS_interval,
            # step_turning_angle
            all_weights = []
            all_bearing_angle = []
            hist = np.ones(len(edges_bearing) - 1)
            for track in cur_cond.accepted_tracks:

                not_nan_bidx = ~np.isnan(cur_cond.bearing_angle) * ~np.isnan(
                    cur_cond.heading_angle)  # track["track_bidx"]

                # not_nan_bidx = not_nan_bidx * track["track_bidx"]

                if (True):

                    if variable_name == 'INS_reorient':
                        bearing_angle = cur_cond.bearing_angle[not_nan_bidx][
                                            cur_cond.step_idx[track["track_step_bidx"]]][:-1]

                        INS_heading_angles = cur_cond.heading_angle[not_nan_bidx][
                            cur_cond.step_idx[track["track_step_bidx"]]]

                        INS_reorient = np.diff(np.abs(INS_heading_angles))

                        # INS_reorient = INS_reorient[cur_cond.next_event_is_step[
                        # track["track_step_bidx"]]]
                        weights = INS_reorient
                    if variable_name == 'run_speed':
                        bearing_angle = self.dict[condition].bearing_angle[track["track_bidx"]]

                        weights = self.dict[condition].midpoint_speed[track["track_bidx"]]

                    # all_bearing_angle.append(np.mean(bearing_angle))
                    # all_weights.append(np.mean(weights))

                    # bearing_angle=np.array(all_bearing_angle).flatten()

                    # weights=(np.array(all_weights)).flatten()

                    # add data for circular boundary conditions
                    bearing_angle = np.hstack(
                        [bearing_angle - 2 *
                         np.pi, bearing_angle, bearing_angle + 2 * np.pi])
                    weights = np.tile(weights, 3)

                    # hist
                    hist += np.nan_to_num(binned_statistic(bearing_angle, weights, bins=edges_bearing)[0])

                    # convolve, filter width = 60 degree
            hist = np.convolve(np.ones(11) / 11., hist, mode='same')
            # plot
            line_data = np.vstack((line_data, hist / len(cur_cond.accepted_tracks)))

        # save data
        if par['save_data']:

            data_xrange = [-180, 180]
            column_names = list(conditions)
            column_names.insert(0, 'Bearing Angle')
            saved_line_idx = np.abs(line_data[0]) <= 183
            title = (str(variable_name) + " to bearing " +
                     dstr)
            short_title = str(variable_name) + '_bear_line' + dstrshort
            df = pandas.DataFrame(line_data.T[saved_line_idx])
            df.columns = column_names

            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title]
            )
            self.excel_titles[short_title] = "LinePlot: " + title

    def figure_variable_depending_on_bearing_distance_split(
            self, par, conditions,
            variable_name, distance,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        if variable_name in ["Abs_HC_angle_turn_TA"]:
            return
        if variable_name in ["Abs_HC_angle_head_TA"]:
            variable_name = "Abs_HC_angle"

        description = {
            'INS_interval': 'Inter-step-interval to bearing',
            'INS_turn': 'Inter-step-turn to bearing',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate to bearing',
            'HCs': 'HCs to bearing',
            'HC_angle': 'HC angle to bearing',
            'Abs_HC_angle': 'Absolute HC angle',
            'run_speed': 'Run Speed to bearing',
            'HC Proportion': 'Proportion of HCs ',
            'HC_Proportion_Reor': 'Proportions of postive/negative Reorientations'
        }

        dstr = description[variable_name]
        dstrshort = str(variable_name)

        # ylim = {
        #     'INS_interval': (0.7, 0.8),
        #     'INS_turn': (-3, 3),
        #     'INS_distance': (0.7, 0.8),
        #     'HC_rate': (0.15, 0.27),
        #     'HC_angle': (-10, 10),
        #     'run_speed': (0.4, 0.6)
        #     }

        # yticks = {
        #     'INS_interval': [0.7, 0.75, 0.8],
        #     'INS_turn': [-3, 0, 3],
        #     'INS_distance': [0.7, 0.75, 0.8],
        #     'HC_rate': [0.15, 0.2, 0.25],
        #     'HC_angle': [-10, 0, 10],
        #     'run_speed': [0.4, 0.5, 0.6]
        #     }

        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        # for all conditions
        # print conditions
        for condition in conditions:

            dstr = description[variable_name]
            dstrshort = str(variable_name)

            cur_cond = copy.copy(self.dict[condition])

            # mean_INS_distance, mean_INS_interval,
            # step_turning_angle
            if variable_name in ['INS_distance', 'INS_interval',
                                 'INS_turn']:

                near_next_event_step_idx = cur_cond.next_event_is_step[
                    cur_cond.distance[cur_cond.step_idx[
                        cur_cond.next_event_is_step]] < distance
                    ]
                far_next_event_step_idx = cur_cond.next_event_is_step[
                    cur_cond.distance[cur_cond.step_idx[
                        cur_cond.next_event_is_step]] >= distance
                    ]
                near_bearing_angle = cur_cond.bearing_angle[
                    cur_cond.step_idx[near_next_event_step_idx]
                ]
                far_bearing_angle = cur_cond.bearing_angle[
                    cur_cond.step_idx[far_next_event_step_idx]
                ]

                near_weights = getattr(
                    cur_cond,
                    variable_name)[near_next_event_step_idx]
                far_weights = getattr(
                    cur_cond,
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
                    far_bearing_angle = far_bearing_angle[far_idx_ok]
                    near_weights = near_weights[near_idx_ok]
                    far_weights = far_weights[far_idx_ok]

            # HC rate
            if variable_name in ['HCs',
                                 'HC_rate']:
                if subthreshold:
                    dstr = dstr + "(max angle " + str(large_HC) + ")"
                    dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                else:
                    dstr = dstr + "(min angle " + str(large_HC) + ")"
                    dstrshort = dstrshort + "_minA" + str(int(large_HC))

                # FIX FILTER HERE
                large_HC_idx = mf.angleComp(self.dict[condition].HC_angle,
                                         large_HC,
                                         subthreshold)

                idx_not_nan = ~np.isnan(
                    cur_cond.HC_initiation)
                near_idx_not_nan = (
                        idx_not_nan & (np.nan_to_num(
                    cur_cond.distance - distance) < 0))
                far_idx_not_nan = (
                        idx_not_nan & (np.nan_to_num(
                    cur_cond.distance - distance) > 0))

                filtered_HC_rate = np.zeros(
                    self.dict[condition].HC_initiation.shape)
                filtered_HC_rate[
                    self.dict[condition].HC_start_idx[large_HC_idx]] = 1

                # print idx_not_nan.shape
                near_bearing_angle = cur_cond.bearing_angle[near_idx_not_nan]
                far_bearing_angle = cur_cond.bearing_angle[far_idx_not_nan]
                near_weights = filtered_HC_rate[
                                   near_idx_not_nan] / float(par['dt'])
                far_weights = filtered_HC_rate[
                                  far_idx_not_nan] / float(par['dt'])

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(cur_cond.midpoint_speed)
                idx_non_hc = cur_cond.HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                idx_non_hc = idx_non_hc * idx_not_nan
                near_idx_non_hc = (
                        idx_non_hc & (np.nan_to_num(
                    cur_cond.distance - distance) < 0))
                far_idx_non_hc = (
                        idx_non_hc & (np.nan_to_num(
                    cur_cond.distance - distance) > 0))
                near_bearing_angle = cur_cond.bearing_angle[
                    near_idx_non_hc]
                far_bearing_angle = cur_cond.bearing_angle[
                    far_idx_non_hc]
                # weights=cur_cond.dict[cur_cond.full_condition].centroid_speed[
                #     idx_non_hc]
                near_weights = cur_cond.midpoint_speed[
                    near_idx_non_hc]
                far_weights = cur_cond.midpoint_speed[
                    far_idx_non_hc]

            # HC angle
            if variable_name in ['HC_angle', 'Abs_HC_angle']:
                if subthreshold:
                    dstr = dstr + "(max angle " + str(large_HC) + ")"
                    dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                else:
                    dstr = dstr + "(min angle " + str(large_HC) + ")"
                    dstrshort = dstrshort + "_minA" + str(int(large_HC))

                near_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]
                near_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_large_HC_idx = mf.angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = mf.angleComp(far_HC_angle,
                                             large_HC,
                                             subthreshold)

                near_bearing_angle = cur_cond.bearing_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_bearing_angle = cur_cond.bearing_angle[
                    far_HC_start_idx[far_large_HC_idx]]
                if variable_name in ['Abs_HC_angle']:
                    near_weights = np.fabs(np.rad2deg(
                        near_HC_angle[near_large_HC_idx]))
                    far_weights = np.fabs(np.rad2deg(
                        far_HC_angle[far_large_HC_idx]))
                else:
                    near_weights = np.rad2deg(
                        near_HC_angle[near_large_HC_idx])
                    far_weights = np.rad2deg(
                        far_HC_angle[far_large_HC_idx])
            if variable_name in ['HC Proportion']:
                near_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_large_HC_idx = mf.angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = mf.angleComp(far_HC_angle,
                                             large_HC,
                                             subthreshold)
                near_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_bearing_angle = self.dict[condition].bearing_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_bearing_angle = self.dict[condition].bearing_angle[
                    far_HC_start_idx[far_large_HC_idx]]

                near_weights = np.rad2deg(
                    near_HC_angle[near_large_HC_idx])
                far_weights = np.rad2deg(
                    far_HC_angle[far_large_HC_idx])

                idx_towards_near = near_bearing_angle * near_weights < 0.
                idx_towards_far = far_bearing_angle * far_weights < 0.
                idx_away_near = near_bearing_angle * near_weights > 0.
                idx_away_far = far_bearing_angle * far_weights > 0.
                far_weights = idx_towards_far.astype(int)
                near_weights = idx_towards_near.astype(int)

            if (variable_name == 'HC_Proportion_Reor'):
                near_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_angle = cur_cond.HC_angle[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_large_HC_idx = mf.angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = mf.angleComp(far_HC_angle,
                                             large_HC,
                                             subthreshold)

                near_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_start_idx = cur_cond.HC_start_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_HC_end_idx = cur_cond.HC_end_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] < distance]
                far_HC_end_idx = cur_cond.HC_end_idx[
                    cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                near_bearing_angle = self.dict[condition].bearing_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_bearing_angle = self.dict[condition].bearing_angle[
                    far_HC_start_idx[far_large_HC_idx]]

                near_heading_angle_at_start = cur_cond.heading_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_heading_angle_at_start = cur_cond.heading_angle[
                    far_HC_start_idx[far_large_HC_idx]]

                near_heading_angle_at_end = cur_cond.heading_angle[
                    near_HC_end_idx[near_large_HC_idx]]
                far_heading_angle_at_end = cur_cond.heading_angle[
                    far_HC_end_idx[far_large_HC_idx]]

                near_weights = np.rad2deg(
                    np.abs(near_heading_angle_at_start) -
                    np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_start) -
                    np.abs(far_heading_angle_at_end))

                near_weights = np.array([x > 0 for x in near_weights]).astype(int)
                far_weights = np.array([x > 0 for x in far_weights]).astype(int)

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

            if (variable_name != 'HCs'):
                near_hist = near_hist / near_n_samples
                far_hist = far_hist / far_n_samples

            if (variable_name in ['HC Proportion', 'HC_Proportion_Reor']):
                far_hist = binned_statistic(far_bearing_angle, far_weights, bins=edges_bearing)[0]
                near_hist = binned_statistic(near_bearing_angle, near_weights, bins=edges_bearing)[0]

            # convolve, filter width = 60 degree
            near_hist = np.convolve(np.ones(11) / 11., near_hist, mode='same')
            far_hist = np.convolve(np.ones(11) / 11., far_hist, mode='same')
            line_data = np.vstack((line_data, near_hist))
            line_data = np.vstack((line_data, far_hist))

        # save data
        if par['save_data']:
            data_xrange = [-180, 180]
            cur_column_names = [
                x + '/' + str(y) for x in list(conditions) for y in [
                    'near', 'far']]
            cur_column_names.insert(0, 'Bearing Angle')
            saved_line_idx = np.abs(line_data[0]) <= 183
            df = pandas.DataFrame(line_data.T[saved_line_idx])
            df.columns = cur_column_names
            title = dstr + " split distance"
            short_title = dstrshort + '_bear_line_d'
            df.to_excel(
                self.excelWriter,
                sheet_name=dstrshort + '_bear_line_d',
                index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    dstrshort + '_bear_line_d'])
            self.excel_titles[short_title] = "LinePlot: " + title

    def figure_boxplot_PREF(self, par, conditions):
        print("figure_boxplot_PREF: ", flush=True)
        # init
        pref = []
        pref_data = []
        count_high = 0
        count_medium = 0
        count_low = 0
        track_length_high = 0.
        track_length_medium = 0.
        track_length_low = 0.
        pref_limit = par['preference_limit']

        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # print cond.odor_A
            # for all accepted tracks
            for track in cond.accepted_tracks:
                # Angle to random shuffle tracks to test for PREF test shape
                # angle = np.pi * np.random.rand(1, 1)[0][0]
                not_nan_bidx = ~np.isnan(cond.spine4[:, 1])
                not_nan_bidx = not_nan_bidx * track["track_bidx"]
                spine4 = 1.0 * cond.spine4[not_nan_bidx]
                # track_dist = cond.distance[not_nan_bidx]
                # print spine4.shape
                # spine4 = np.array(rotate_vector_clockwise(angle, spine4.T)).T
                # print spine4.shape
                # init
                t_up = 0.
                t_down = 0.

                # spine4_y
                spine4_y = spine4[:, 1]

                # inserted by Michael for test 08.11.2018
                spine4_x = spine4[:, 0]

                # time up and down
                t_down = np.sum(spine4_y < 0.)
                t_up = np.sum(spine4_y > 0.)
                # limit distance:
                # limit = track["trial_radius"]
                # limit = par['d_split']/1.2
                # t_up = np.sum(track_dist < limit)
                # t_down = np.sum(track_dist > limit)
                avg_dist = np.nanmean(cond.distance[track["track_bidx"]])
                avg_dist_norm = 1.0 - (avg_dist / par['d_split'])
                PREF_VAL = (t_up - t_down) / (t_down + t_up).astype(float)
                pref_data.append([
                    condition,
                    track["trial"],
                    track["track_number"],
                    avg_dist,
                    avg_dist_norm,
                    PREF_VAL,
                    len(spine4_y)])
                pref.append(PREF_VAL)
                track['PREF'] = PREF_VAL
                if PREF_VAL >= pref_limit:
                    count_high = count_high + 1.
                    track_length_high = track_length_high + np.nansum(
                        ~np.isnan(cond.time[track["track_bidx"]]))
                elif PREF_VAL <= -1. * pref_limit:
                    count_low = count_low + 1.
                    track_length_low = track_length_low + np.nansum(
                        ~np.isnan(cond.time[track["track_bidx"]]))
                else:
                    count_medium = count_medium + 1.
                    track_length_medium = track_length_medium + np.nansum(
                        ~np.isnan(cond.time[track["track_bidx"]]))

        print("Avg Track length high: " + str(track_length_high / count_high))
        print("Avg Track length low: " + str(track_length_low / count_low))
        print("Avg Track length medium: " + str(track_length_medium / count_medium))

        # convert to array
        pref = np.array(pref)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(pref_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Track",
                          "AvgDist",
                          "AvgDistNormed",
                          "PREF", "N"]
            df.to_excel(self.excelWriter,
                        sheet_name='PREF',
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'PREF'])


            df_high = df[df['PREF'] > par['preference_limit']]
            df_low = df[df['PREF'] < (-1.0 * par['preference_limit'])]
            df_mod = df[np.abs(df['PREF']) <= par['preference_limit']]
            # xdistance_range= (
            #     np.nanmin(df['AvgDist']), np.nanmax(df['AvgDist']))
            xdistance_range = (0, 2 * par['radius_dish'])
            for condition in conditions:

                # custom histogram
                chist = np.histogram(df[df["Group"] == condition]["PREF"],
                                     bins=[-1, -0.9, 0, 0.9, 1],
                                     range=(-1.0, 1.0)
                                     )
                chistv = 100.0 * chist[0] / np.sum(chist[0])
                dfb = pandas.DataFrame(chistv)
                dfb.columns = ["P"]
                dfb["Group"] = str(condition)
                dfb["L"] = chist[1][:-1] + (np.diff(chist[1]) / 2.0)

            self.excel_titles['PREF'] = 'PREFs'

    def figure_HC_reorientation_boxplot(
            self,
            par,
            conditions,
            distance,
            reorientation=True,
            large_HC=-1,
            subthreshold=False,
            bearing_limited='nl'
    ):
        if (large_HC == -1):
            large_HC = par['large_HC']

        lsuffix = ''
        long_lsuffix = ''
        if (bearing_limited == 'l'):
            lsuffix = "_l"
            long_lsuffix = 'limited'
        if (bearing_limited == 'il'):
            lsuffix = "_il"
            long_lsuffix = 'inverse limited'

        if reorientation:
            dstr = 'reor'
            ylabel = "HC reorientation"
            yrange = (-30, 30)
            yt = np.arange(-30, 31, 5)
            hline_at = 0
        else:
            dstr = 'accuracy'
            ylabel = "HC accuracy"
            yrange = (60, 120)
            yt = np.arange(60, 121, 10)
            hline_at = 90

        if subthreshold:
            angle_str = " (max angle:" + str(large_HC) + ") "
            dstrshort = dstr + "_maxA" + str(int(large_HC))
            dstr = dstr + angle_str
        else:
            angle_str = " (min angle:" + str(large_HC) + ")  "
            dstrshort = dstr + "_minA" + str(int(large_HC))
            dstr = dstr + angle_str

        boxplot_data = []
        for condition in conditions:
            cur_cond = self.dict[condition]
            for track in cur_cond.accepted_tracks:

                pref_trials = [
                    cur_cond.trial[index] for index in sorted(
                        np.unique(cur_cond.trial, return_index=True)[1])
                ]
                trial_numbers = [
                    cur_cond.trial_number[index] for index in sorted(
                        np.unique(
                            cur_cond.trial_number, return_index=True)[1])
                ]
                trial_dict = dict(list(zip(trial_numbers, pref_trials)))
                full_condition = cur_cond.full_condition

                HC_start_idx = cur_cond.HC_start_idx[track["track_HC_bidx"]]

                HC_end_idx = cur_cond.HC_end_idx[track["track_HC_bidx"]]

                HC_angle = cur_cond.HC_angle[track["track_HC_bidx"]]

                if (bearing_limited == 'l' or bearing_limited == 'il'):
                    bearing_angle_idx = np.array([x <= np.deg2rad(135) and x >= np.deg2rad(45)
                                                  or x >= np.deg2rad(-135) and x <= np.deg2rad(-45) for x in
                                                  cur_cond.bearing_angle])[
                        cur_cond.HC_start_idx]
                    if (bearing_limited == 'il'):
                        bearing_angle_idx = ~bearing_angle_idx
                    HC_start_idx = HC_start_idx[bearing_angle_idx]
                    HC_end_idx = HC_end_idx[bearing_angle_idx]
                    HC_angle = HC_angle[bearing_angle_idx]

                large_HC_bidx = mf.angleComp(HC_angle, large_HC,
                                          subthreshold)

                heading_angle_at_start = cur_cond.heading_angle[
                    HC_start_idx[large_HC_bidx]]
                heading_angle_at_end = cur_cond.heading_angle[
                    HC_end_idx[large_HC_bidx]]

                if reorientation:

                    weights = np.rad2deg(
                        np.abs(heading_angle_at_start) -
                        np.abs(heading_angle_at_end))

                else:

                    weights = np.rad2deg(
                        np.abs(heading_angle_at_end))

                weight_average = (
                        np.sum(weights) / len(weights)
                )

                boxplot_data.append([
                    condition,
                    track["trial"],
                    track["track_id"],
                    weight_average,
                ])

        # save data
        if par['save_data']:

            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial",
                          'Track ID',
                          'value']
            if reorientation:
                description = """
                HC reorientation is the
                abs(Heading angle at start of HC)-
                abs(Heading angle at end of HC).
                It shows how much the bearing is
                potentially improved by each HC.
                Average for each trial is a data point for the boxplot.
                """
                base = 0.0

            else:
                description = """
                HC Accuracy is the abs(HC angle at end of HC). How well
                directed is each HC. Average for each trial
                is a data point for the boxplot.
                """
                base = 90.0

            title = "Boxplot: HC " + dstr + long_lsuffix
            short_title = 'HC_' + dstrshort + '_box' + lsuffix

            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])
            self.excel_titles[short_title] = title

    def figure_proportion_of_time_boxplot(
            self,
            par,
            conditions
    ):
        print("figure_proportion_of_time")
        title = "Proportion of time towards odour"
        short_title = 'prop_time_to_away_box'
        column_names = ['To odor', 'Away from odor']
        # column_names_short = ['towards', 'away']

        # init
        boxplot_data = []
        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # for all accepted tracks
            for track in cond.accepted_tracks:
                not_nan_bidx = ~np.isnan(cond.bearing_angle)
                not_nan_bidx = not_nan_bidx * track["track_bidx"]

                weights = cond.bearing_angle[not_nan_bidx]

                if (len(weights) < par['ind_lim_prop']):
                    continue
                idx_twrd = np.sum(np.abs(weights) < (np.pi / 2))
                idx_away = np.sum(np.abs(weights) > (np.pi / 2))
                proportion_twrd = np.nan
                if (idx_away + idx_twrd != 0):
                    proportion_twrd = (
                            float(idx_twrd) / float(idx_away + idx_twrd))

                if np.isnan(proportion_twrd) | np.isnan(proportion_twrd):
                    continue

                boxplot_data.append([
                    condition,
                    track["trial"],
                    track["track_number"],
                    proportion_twrd,
                    idx_twrd + idx_away,
                    track["PREF"]
                ])
        print(boxplot_data)
        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Track Name",
                          column_names[0],
                          "N",
                          "PREF"]



            df_ex = df[np.abs(df['PREF']) > par['preference_limit']]
            df_ex['PREF Class'] = "None"
            df_ex.loc[
                df_ex['PREF'] > par['preference_limit'], 'PREF Class'] = 'High'
            df_ex.loc[
                df_ex['PREF'] < par['preference_limit'], 'PREF Class'] = 'Low'
            df_high = df[df['PREF'] > par['preference_limit']]
            df_low = df[df['PREF'] < (-1 * par['preference_limit'])]
            df_mod = df[np.abs(df['PREF']) <= par['preference_limit']]
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)

            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    short_title])
            self.excel_titles[short_title] = title

    def figure_proportion_of_time_boxplot_distance_split(
            self,
            par,
            conditions,
            distance
    ):
        print("figure_proportion_of_time_boxplot_distance_split", flush=True)
        title = "Proportion of time towards odour"
        short_title = 'prop_time_to_away_box'
        column_names = ['Proportion To', 'Proportion Away']

        # init
        boxplot_data = []
        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # for all accepted tracks
            for track in cond.accepted_tracks:
                not_nan_bidx = ~np.isnan(cond.bearing_angle)
                not_nan_bidx = not_nan_bidx * track["track_bidx"]
                near_bidx = cond.distance < distance
                far_bidx = cond.distance >= distance
                not_nan_bidx_near = near_bidx * not_nan_bidx
                not_nan_bidx_far = far_bidx * not_nan_bidx

                if (any(not_nan_bidx_near)):
                    near_weights = cond.bearing_angle[not_nan_bidx_near]
                else:
                    near_weights = np.array([])

                if (any(not_nan_bidx_far)):
                    far_weights = cond.bearing_angle[not_nan_bidx_far]
                else:
                    far_weights = np.array([])

                if (len(near_weights) > 0):
                    count_near = np.sum(np.sum(
                        np.abs(near_weights) < par['to_range']))
                    proportion_near = float(count_near) / len(near_weights)
                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        "Near",
                        proportion_near,
                        len(near_weights),
                        track["PREF"]
                    ])

                if (len(far_weights) > 0):
                    count_far = np.sum(
                        np.sum(np.abs(far_weights) < par['to_range']))
                    proportion_far = float(count_far) / len(far_weights)
                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        "Far",
                        proportion_far,
                        len(far_weights),
                        track["PREF"]
                    ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Track Name",
                          "Near/Far",
                          column_names[0],
                          "N",
                          "PREF"]

            # Michael 05.12.2018
            df_ex = df[np.abs(df['PREF']) > par['preference_limit']]
            df_ex['PREF Class'] = "None"
            df_ex.loc[
                df_ex['PREF'] > par['preference_limit'], 'PREF Class'] = 'High'
            df_ex.loc[
                df_ex['PREF'] < par['preference_limit'], 'PREF Class'] = 'Low'
            # Only those with moderate PREF
            df_high = df[df['PREF'] > par['preference_limit']]
            df_low = df[df['PREF'] < (-1 * par['preference_limit'])]
            df_mod = df[np.abs(df['PREF']) <= par['preference_limit']]


            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)

            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])
            self.excel_titles[short_title] = title

    def figure_proportion_of_HCs_boxplot(
            self,
            par
    ):
        print("figure_proportion_of_HCs_boxplot", flush=True)
        ylabel = "Proportion of HCs"
        column_names = ['To odor', 'Away from odor']
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

        large_HC_idx = np.abs(
            self.HC_angle) > np.deg2rad(par['large_HC'])
        bearing_angle = self.bearing_angle[
            self.HC_start_idx[large_HC_idx]]

        track_numbers = [
            self.track_number[index] for index in sorted(
                np.unique(self.track_number,
                          return_index=True)[1])
        ]
        # for all trials
        for trial_number in trial_numbers:

            idx_trial = (
                    self.trial_number
                    [self.HC_start_idx[large_HC_idx]] ==
                    trial_number)

            for track_number in track_numbers:
                idx_track = (
                        self.track_number[self.HC_start_idx[large_HC_idx]] ==
                        track_number)
                idx_trial_track = idx_track & idx_trial
                if (not any(idx_trial_track)):
                    continue
                c_bearing_angle = bearing_angle[idx_trial_track]

                weights = np.rad2deg(
                    self.HC_angle[large_HC_idx])

                weights = weights[idx_trial_track]
                if (len(weights) < par['ind_lim_prop']):
                    continue
                idx_twrd = np.sum(c_bearing_angle * weights < 0.)
                idx_away = np.sum(c_bearing_angle * weights > 0.)
                proportion_twrd = np.nan
                proportion_away = np.nan
                if (idx_away + idx_twrd != 0):
                    proportion_twrd = (
                            float(idx_twrd) / float(idx_away + idx_twrd))
                    proportion_away = (
                            float(idx_away) / float(idx_away + idx_twrd))

                if np.isnan(proportion_twrd) | np.isnan(proportion_twrd):
                    continue

                boxplot_black.append(proportion_twrd)
                boxplot_white.append(proportion_away)
                boxplot_data.append([
                    full_condition,
                    trial_dict[trial_number],
                    track_number,
                    proportion_twrd,
                    proportion_away,
                    idx_twrd + idx_away
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
                          "Track Name",
                          column_names[0],
                          column_names[1],
                          "N"]
            df.to_excel(self.excelWriter,
                        sheet_name='proportions_to_away_box',
                        index=False)

            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'proportions_to_away_box'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        'proportions_to_away_boxplot')
            plt.close()

    def figure_proportion_of_HCs_boxplot_distance_split(
            self,
            par,
            distance
    ):
        print("figure_proportion_of_HCs_boxplot_distance_split", flush=True)

        ylabel = "Proportion of HCs"
        column_names = ['Near', 'Far']

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

        near_HC_bidx = self.distance[self.HC_start_idx] < distance
        far_HC_bidx = self.distance[self.HC_start_idx] >= distance
        large_HC_bidx = np.abs(
            self.HC_angle) > np.deg2rad(par['large_HC'])
        near_large_HC_bidx = near_HC_bidx * large_HC_bidx
        far_large_HC_bidx = far_HC_bidx * large_HC_bidx
        near_large_HC_bearing_angle = self.bearing_angle[
            self.HC_start_idx[near_large_HC_bidx]]
        far_large_HC_bearing_angle = self.bearing_angle[
            self.HC_start_idx[far_large_HC_bidx]]

        track_numbers = [
            self.track_number[index] for index in sorted(
                np.unique(self.track_number,
                          return_index=True)[1])
        ]
        # for all trials
        for trial_number in trial_numbers:

            near_bidx_trial = (
                    self.trial_number
                    [self.HC_start_idx[near_large_HC_bidx]] ==
                    trial_number)
            far_bidx_trial = (
                    self.trial_number
                    [self.HC_start_idx[far_large_HC_bidx]] ==
                    trial_number)

            for track_number in track_numbers:
                near_bidx_track = (
                        self.track_number[self.HC_start_idx[near_large_HC_bidx]] ==
                        track_number)
                near_bidx_trial_track = near_bidx_track & near_bidx_trial

                far_bidx_track = (
                        self.track_number[self.HC_start_idx[far_large_HC_bidx]] ==
                        track_number)
                far_bidx_trial_track = far_bidx_track & far_bidx_trial

                if (any(near_bidx_trial_track)):
                    near_c_bearing_angle = near_large_HC_bearing_angle[
                        near_bidx_trial_track]
                    near_weights = np.rad2deg(
                        self.HC_angle[near_large_HC_bidx])
                    near_weights = near_weights[near_bidx_trial_track]
                else:
                    near_c_bearing_angle = np.array([])
                    near_weights = np.array([])

                if (any(far_bidx_trial_track)):
                    far_c_bearing_angle = far_large_HC_bearing_angle[
                        far_bidx_trial_track]
                    far_weights = np.rad2deg(
                        self.HC_angle[far_large_HC_bidx])
                    far_weights = far_weights[far_bidx_trial_track]
                else:
                    far_c_bearing_angle = np.array([])
                    far_weights = np.array([])

                if (len(near_c_bearing_angle * near_weights) > 0):
                    count_near = np.sum(
                        near_c_bearing_angle * near_weights < 0.)
                    proportion_near = float(count_near) / len(
                        near_c_bearing_angle * near_weights)
                    boxplot_black.append(proportion_near)
                    boxplot_data.append([
                        full_condition,
                        trial_dict[trial_number],
                        track_number,
                        "Near",
                        proportion_near,
                        len(near_weights)
                    ])
                else:
                    count_near = np.nan
                    proportion_near = np.nan

                if (len(far_c_bearing_angle * far_weights) > 0):
                    count_far = np.sum(far_c_bearing_angle * far_weights < 0.)
                    proportion_far = float(count_far) / len(
                        far_c_bearing_angle * far_weights)
                    boxplot_white.append(proportion_far)
                    boxplot_data.append([
                        full_condition,
                        trial_dict[trial_number],
                        track_number,
                        "Far",
                        proportion_far,
                        len(far_weights)
                    ])
                else:
                    proportion_far = np.nan
                    count_far = np.nan

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
                          "Track Name",
                          "Near/Far",
                          "Proportion Towards",
                          "Total HCs"]
            df.to_excel(self.excelWriter,
                        sheet_name='proportions_to_away_box_d',
                        index=False)

            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'proportions_to_away_box_d'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' + par['group_name'] + '_' +
                        '_' + par['condition'] + '_' +
                        'proportions_to_away_boxplot_d')
            plt.close()

    def figure_boxplot_variable_depending_on_parameter(
            self,
            par,
            conditions,
            variable_name,
            subthreshold=False,
            parameter='bearing',
            plot_total=False,
            large_HC=-1):
        print("figure_boxplot_variable_depending_on_parameter", flush=True)
        if (large_HC == -1):
            large_HC = par['large_HC']

        # figure settings
        description = {
            'INS_interval': 'Inter-step-interval',
            'INS_turn': 'Inter-step-turn',
            'INS_distance': 'Inter-step-distance (mm)',
            'INS_reorient': 'Inter-step-reorientation',
            'HC_rate': 'HC rate',
            'HC_angle': 'HC angle',
            'Abs_HC_angle_turn_TA': 'Absolute HC angle when turning' +
                                    ' to/away from the odour',
            'Abs_HC_angle_head_TA': 'Absolute HC angle when heading' +
                                    ' to/away from the odour',
            'run_speed': 'Run Speed',
            'HC_reorientation': 'HC Reorientation',
            'HC_accuracy': 'HC Accuracy',
        }

        param_column = {
            'distance': ['near', 'far'],
            'bearing': ['toward', 'away'],
            'time': ['1_half', '2_half'],
        }

        column_names = {
            'INS_interval': ['toward', 'away'],
            'INS_turn': ['left', 'right'],
            'INS_distance': ['toward', 'away'],
            'HC_angle': ['left', 'right'],
            'Abs_HC_angle_turn_TA': ['turn towards', 'turn away'],
            'Abs_HC_angle_head_TA': ['toward', 'away'],
        }

        measure = {
            'INS_interval': 'Steps',
            'INS_turn': 'Steps',
            'INS_distance': 'Steps',
            'INS_reorient': 'Steps',
            'HC_rate': 'HCs',
            'HC_angle': 'HCs',
            'Abs_HC_angle_turn_TA': 'HCs',
            'Abs_HC_angle_head_TA': 'HCs',
            'run_speed': 'Run Frames',
            'HC_reorientation': 'HCs',
            'HC_accuracy': 'HCs',
        }

        # init
        parameter_modulation_data = []
        boxplot_data = []
        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # for all accepted tracks
            for track in cond.accepted_tracks:
                dstr = description[variable_name]
                dstrshort = str(variable_name)
                # HC rate
                if variable_name == 'HC_rate':
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    large_HC_bidx = mf.angleComp(cond.HC_angle,
                                              large_HC,
                                              subthreshold)
                    idx_not_nan = ~np.isnan(cond.HC_initiation)
                    large_track_HC_bidx = large_HC_bidx * track["track_HC_bidx"]
                    filtered_HC_rate = np.zeros(
                        cond.HC_initiation.shape)
                    filtered_HC_rate[
                        cond.HC_start_idx[large_track_HC_bidx]] = 1
                    bearing_angle = cond.bearing_angle[
                        idx_not_nan * track["track_bidx"]]
                    time = cond.time[
                        idx_not_nan * track["track_bidx"]]
                    distance = cond.distance[
                        idx_not_nan * track["track_bidx"]]
                    weights = filtered_HC_rate / float(par['dt'])
                    weights = weights[
                        idx_not_nan * track["track_bidx"]]
                    if len(weights) == 0:
                        continue

                # Run Speed
                if variable_name == 'run_speed':
                    idx_not_nan = ~np.isnan(cond.HC_initiation)
                    bearing_angle = cond.bearing_angle[
                        idx_not_nan * track["track_bidx"]]
                    time = cond.time[
                        idx_not_nan * track["track_bidx"]]
                    distance = cond.distance[
                        idx_not_nan * track["track_bidx"]]
                    weights = cond.midpoint_speed[
                        idx_not_nan * track["track_bidx"]]

                # Inter Step reorientation
                if variable_name == 'INS_reorient':
                    INS_heading_angles = cond.heading_angle[
                        cond.step_idx[track["track_step_bidx"]]]
                    INS_reorient = np.diff(np.abs(INS_heading_angles))
                    INS_reorient = INS_reorient[cond.next_event_is_step[
                                                    track["track_step_bidx"]][:-1]]
                    weights = INS_reorient
                    bearing_angle = cond.bearing_angle[
                        cond.step_idx[track["track_step_bidx"]]]
                    bearing_angle = bearing_angle[cond.next_event_is_step[
                        track["track_step_bidx"]]]
                    time = cond.time[
                        cond.step_idx[track["track_step_bidx"]]]
                    time = time[cond.next_event_is_step[
                        track["track_step_bidx"]]]
                    distance = cond.distance[
                                   cond.step_idx[track["track_step_bidx"]]][:-1]
                    distance = distance[cond.next_event_is_step[
                                            track["track_step_bidx"]][:-1]]

                # HC reorientation
                if variable_name in ['HC_reorientation', 'HC_accuracy']:

                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    large_HC_bidx = mf.angleComp(cond.HC_angle,
                                              large_HC,
                                              subthreshold)
                    large_track_HC_bidx = large_HC_bidx * track["track_HC_bidx"]

                    bearing_angle = cond.bearing_angle[
                        cond.HC_start_idx[large_track_HC_bidx]]
                    time = cond.time[
                        cond.HC_start_idx[large_track_HC_bidx]]
                    distance = cond.distance[
                        cond.HC_start_idx[large_track_HC_bidx]]

                    HC_start_idx = cond.HC_start_idx[large_track_HC_bidx]
                    HC_end_idx = cond.HC_end_idx[large_track_HC_bidx]

                    heading_angle_at_start = cond.heading_angle[HC_start_idx]
                    heading_angle_at_end = cond.heading_angle[HC_end_idx]

                    if variable_name == 'HC_reorientation':
                        weights = np.rad2deg(np.abs(heading_angle_at_start) -
                                             np.abs(heading_angle_at_end))
                        if len(weights) == 0:
                            continue
                    else:
                        weights = np.rad2deg(np.abs(heading_angle_at_end))

                # HC angle
                if variable_name in ['HC_angle',
                                     'Abs_HC_angle_turn_TA',
                                     'Abs_HC_angle_head_TA']:

                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    large_HC_bidx = mf.angleComp(cond.HC_angle,
                                              large_HC,
                                              subthreshold)
                    large_track_HC_bidx = large_HC_bidx * track["track_HC_bidx"]

                    bearing_angle = cond.bearing_angle[
                        cond.HC_start_idx[large_track_HC_bidx]]
                    time = cond.time[
                        cond.HC_start_idx[large_track_HC_bidx]]
                    distance = cond.distance[
                        cond.HC_start_idx[large_track_HC_bidx]]

                    weights = np.rad2deg(cond.HC_angle[large_track_HC_bidx])
                    if len(weights) == 0:
                        continue

                if variable_name in ['INS_distance',
                                     'INS_interval',
                                     'Abs_HC_angle_turn_TA']:
                    idx_black = np.abs(bearing_angle) < par['to_range']
                    idx_white = np.abs(bearing_angle) > par['away_range']
                    val_black = 0
                    val_white = 0
                    if len(weights) >= par['ind_lim_' + variable_name]:
                        variance_black = np.nanvar(weights[idx_black])
                        variance_white = np.nanvar(weights[idx_white])
                        variance_total = np.nanvar(weights)
                        val_black = np.nanmean(weights[idx_black])
                        val_white = np.nanmean(weights[idx_white])
                        val_total = np.nanmean(weights)
                    else:
                        continue

                    if np.isnan(val_black):
                        continue
                    if np.isnan(val_white):
                        continue
                    if np.isnan(val_total):
                        continue

                    ncount_black = 0
                    ncount_white = 0
                    if variable_name in ['Abs_HC_angle_turn_TA']:
                        variance_black = np.nanvar(np.abs(weights[idx_black]))
                        variance_white = np.nanvar(np.abs(weights[idx_white]))
                        variance_total = np.nanvar(np.abs(weights))
                        val_black = np.mean(np.abs(weights[idx_black]))
                        val_white = np.mean(np.abs(weights[idx_white]))
                        val_total = np.mean(np.abs(weights))
                        ncount_black = np.sum(idx_black)
                        ncount_white = np.sum(idx_white)

                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' ' +
                        column_names[variable_name][0],
                        val_black,
                        ncount_black,  # Number of HCs towards
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_black
                    ])

                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' ' +
                        column_names[variable_name][1],
                        val_white,
                        ncount_white,  # Number of HCs away
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_white
                    ])

                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' total',
                        val_white,
                        ncount_white,  # Number of HCs away
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_total
                    ])

                if variable_name in ['HC_rate', 'run_speed',
                                     'HC_reorientation', 'HC_accuracy',
                                     'INS_reorient', ]:
                    if parameter == "bearing":
                        idx_black = np.abs(bearing_angle) < par['to_range']
                        idx_white = np.abs(bearing_angle) > par['away_range']
                    elif parameter == "distance":
                        idx_black = distance < par['d_split']  # near
                        idx_white = distance > par['d_split']  # far
                    elif parameter == "time":
                        idx_black = time < (par['start_time'] + (
                                (par['end_time'] - par['start_time']) / 2.0))
                        idx_white = time > (par['start_time'] + (
                                (par['end_time'] - par['start_time']) / 2.0))

                    val_black = 0
                    val_white = 0
                    val_total = 0
                    if len(weights) >= par['ind_lim_' + variable_name]:
                        val_black = np.nanmean(weights[idx_black])
                        val_white = np.nanmean(weights[idx_white])
                        val_total = np.nanmean(weights)
                    else:
                        continue

                    ncount_black = 0
                    ncount_white = 0
                    if variable_name in ['HC_rate']:
                        ncount_black = np.sum(weights[idx_black] > 0.0)
                        ncount_white = np.sum(weights[idx_white] > 0.0)
                        ncount_total = np.sum(weights > 0.0)
                        variance_black = np.var(
                            np.diff(np.where(weights[idx_black] > 0)))
                        variance_white = np.var(
                            np.diff(np.where(weights[idx_white] > 0)))
                        variance_total = np.var(
                            np.diff(np.where(weights > 0)))
                    if variable_name in ['HC_reorientation', 'HC_accuracy',
                                         'INS_reorient']:
                        ncount_black = len(weights[idx_black])
                        ncount_white = len(weights[idx_white])
                        ncount_total = len(weights)
                        variance_black = np.var(weights[idx_black])
                        variance_white = np.var(weights[idx_white])
                        variance_total = np.var(weights)
                    if variable_name in ['run_speed']:
                        ncount_black = np.sum(idx_black)
                        ncount_white = np.sum(idx_white)
                        ncount_total = len(weights)
                        variance_black = np.nanvar(weights[idx_black])
                        variance_white = np.nanvar(weights[idx_white])
                        variance_total = np.nanvar(weights)

                    if variable_name in ['HC_reorientation', 'HC_accuracy',
                                         'INS_reorient']:
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][0],
                            val_black,
                            ncount_black,
                            len(weights[idx_black]),
                            track["PREF"],
                            variance_black
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][1],
                            val_white,
                            ncount_white,
                            len(weights[idx_white]),
                            track["PREF"],
                            variance_white
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' total',
                            val_total,
                            ncount_total,
                            len(weights),
                            track["PREF"],
                            variance_total
                        ])
                        b_data = (
                                np.mean(weights[idx_black]) -
                                np.mean(weights[idx_white]))
                        t_data = (
                                np.abs(np.mean(weights[idx_black])) +
                                np.abs(np.mean(weights[idx_white]))
                        )
                        mod_val = 0
                        if t_data is not 0:
                            mod_val = b_data / t_data

                        if not np.isnan(b_data / t_data):
                            parameter_modulation_data.append([
                                condition,
                                track["trial"],
                                track["track_number"],
                                mod_val,
                                track['PREF'],
                                weights.shape[0],
                                weights[idx_black].shape[0],
                                weights[idx_white].shape[0],
                                np.sum(weights >= 0),
                                np.sum(weights[idx_black] >= 0),
                                np.sum(weights[idx_white] >= 0)
                            ])

                    if variable_name in ['run_speed']:
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][0],
                            val_black,
                            ncount_black,
                            len(weights[idx_black]),
                            track["PREF"],
                            variance_black
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][1],
                            val_white,
                            ncount_white,
                            len(weights[idx_white]),
                            track["PREF"],
                            variance_white
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' total',
                            val_total,
                            np.nanmean(weights),
                            ncount_total,
                            track["PREF"],
                            variance_total
                        ])
                        b_data = (
                                np.mean(weights[idx_black]) -
                                np.mean(weights[idx_white]))
                        t_data = (
                                np.mean(weights[idx_black]) +
                                np.mean(weights[idx_white])
                        )

                        if not np.isnan(b_data / t_data):
                            parameter_modulation_data.append([
                                condition,
                                track["trial"],
                                track["track_number"],
                                b_data / t_data,
                                track['PREF'],
                                weights.shape[0],
                                weights[idx_black].shape[0],
                                weights[idx_white].shape[0],
                                np.sum(weights >= 0),
                                np.sum(weights[idx_black] >= 0),
                                np.sum(weights[idx_white] >= 0)
                            ])

                    if variable_name in ['HC_rate']:
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][0],
                            val_black,
                            ncount_black,
                            len(weights[idx_black]),
                            track["PREF"],
                            variance_black
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' ' +
                            param_column[parameter][1],
                            val_white,
                            ncount_white,
                            len(weights[idx_white]),
                            track["PREF"],
                            variance_white
                        ])
                        boxplot_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            condition + ' total',
                            np.nanmean(weights),
                            ncount_total,
                            len(weights),
                            track["PREF"],
                            variance_total
                        ])
                        b_data = (
                                np.mean(weights[idx_white]) -
                                np.mean(weights[idx_black]))
                        t_data = (
                                np.mean(weights[idx_black]) +
                                np.mean(weights[idx_white])
                        )

                        if not np.isnan(b_data / t_data):
                            parameter_modulation_data.append([
                                condition,
                                track["trial"],
                                track["track_number"],
                                b_data / t_data,
                                track['PREF'],
                                weights.shape[0],
                                weights[idx_black].shape[0],
                                weights[idx_white].shape[0],
                                np.sum(weights > 0),
                                np.sum(weights[idx_black] > 0),
                                np.sum(weights[idx_white] > 0)
                            ])

                if variable_name in ['INS_turn', 'HC_angle',
                                     'Abs_HC_angle_head_TA']:
                    idx_black = bearing_angle < 0.
                    idx_white = bearing_angle > 0.
                    val_black = 0
                    val_white = 0
                    if len(weights) >= par['ind_lim_' + variable_name]:
                        variance_black = np.nanvar(weights[idx_black])
                        variance_white = np.nanvar(weights[idx_white])
                        variance_total = np.nanvar(weights)
                        val_black = np.mean(weights[idx_black])
                        val_white = np.mean(weights[idx_white])
                        val_total = np.mean(weights)
                    else:
                        continue

                    if np.isnan(val_black):
                        continue
                    if np.isnan(val_white):
                        continue
                    if np.isnan(val_total):
                        continue

                    ncount_black = 0
                    ncount_white = 0
                    if variable_name in ['INS_turn']:
                        ncount_black = np.count_nonzero(idx_black)
                        ncount_white = np.count_nonzero(idx_white)
                    if variable_name in ['HC_angle']:
                        ncount_black = np.sum(idx_black)
                        ncount_white = np.sum(idx_white)
                    if variable_name in ['Abs_HC_angle_head_TA']:
                        val_black = np.mean(np.abs(weights[idx_black]))
                        val_white = np.mean(np.abs(weights[idx_white]))
                        ncount_black = np.sum(idx_black)
                        ncount_white = np.sum(idx_white)

                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' ' +
                        column_names[variable_name][0],
                        val_black,
                        ncount_black,  # Number of HCs towards
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_black
                    ])
                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' ' +
                        column_names[variable_name][1],
                        val_white,
                        ncount_white,  # Number of HCs away
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_white
                    ])
                    boxplot_data.append([
                        condition,
                        track["trial"],
                        track["track_number"],
                        condition + ' total',
                        val_total,
                        ncount_white,  # Number of HCs away
                        np.sum(track["track_bidx"]),  # Number of Frames
                        track["PREF"],
                        variance_white
                    ])
                    b_data = (
                            np.mean(weights[idx_white]) -
                            np.mean(weights[idx_black]))
                    t_data = (
                            np.mean(weights[idx_black]) +
                            np.mean(weights[idx_white])
                    )

                    if not np.isnan(b_data / t_data):
                        parameter_modulation_data.append([
                            condition,
                            track["trial"],
                            track["track_number"],
                            b_data / t_data,
                            track['PREF'],
                            weights.shape[0],
                            weights[idx_black].shape[0],
                            weights[idx_white].shape[0],
                            np.sum(weights > 0),
                            np.sum(weights[idx_black] > 0),
                            np.sum(weights[idx_white] > 0)
                        ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Track Number",
                          "Criteria",
                          str(variable_name),
                          measure[variable_name],
                          "Frames",
                          "PREF",
                          "Ind. Variance"]

            df_tot = df[df['Criteria'].str.contains('total')]
            df_p = df[~df['Criteria'].str.contains('total')]
            title = dstr
            short_title = dstrshort

            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title]
            )
            self.excel_titles[short_title] = title

            df_high = df_p[df_p['PREF'] > par['preference_limit']]
            df_low = df_p[df_p['PREF'] < (-1.0 * par['preference_limit'])]
            df_mod = df_p[np.abs(df_p['PREF']) <= par['preference_limit']]

            if variable_name in ['HC_rate', 'run_speed',
                                 'HC_reorientation',
                                 'HC_accuracy', 'INS_reorient']:
                pair_by = param_column[parameter]
            else:
                pair_by = column_names[variable_name]


            if variable_name in ['HC_rate', 'run_speed', 'HC_reorientation',
                                 'HC_accuracy']:
                mdf = pandas.DataFrame(parameter_modulation_data)
                mdf.sort_values(by=[0, 1], inplace=True)
                mdf.columns = ["Group",
                               "Trial Name",
                               "Track Number",
                               str(variable_name) + '_modulation_' +
                               parameter,
                               "PREF",
                               "N Frames",
                               "N Frames " + pair_by[0],
                               "N Frames " + pair_by[1],
                               "N HCs",
                               "N HCs " + pair_by[0],
                               "N HCs " + pair_by[1],
                               ]
                # Only those with extreme PREF
                mdf_ex = mdf[np.abs(mdf['PREF']) > par['preference_limit']]
                mdf_ex['PREF Class'] = "None"
                mdf_ex.loc[
                    mdf_ex['PREF'] > par[
                        'preference_limit'], 'PREF Class'] = 'High'
                mdf_ex.loc[
                    mdf_ex['PREF'] < par[
                        'preference_limit'], 'PREF Class'] = 'Low'
                # Only those with moderate PREF
                mdf_high = mdf[mdf['PREF'] > par['preference_limit']]
                mdf_low = mdf[mdf['PREF'] < (-1 * par['preference_limit'])]
                mdf_mod = mdf[
                    np.abs(mdf['PREF']) <= par['preference_limit']]

                title = (dstr +
                         " modulation depending on " + parameter)
                short_title = dstrshort + '_m_to_' + parameter[0]


                mdf.to_excel(self.excelWriter,
                             sheet_name=short_title,
                             index=False)
                mf.fixXLColumns(
                    mdf,
                    self.excelWriter.sheets[short_title
                    ])
                self.excel_titles[short_title] = title

    def figure_boxplot_variable_depending_on_bearing_distance_split(
            self,
            par,
            variable_name,
            distance):

        # this function takes very long to compute, because of ...== trial




        ta = ['toward/near', 'away/near', 'toward/far', 'away/far']
        lr = ['left/near', 'right/near', 'left/far', 'right/far']
        samplesize_names = {
            'INS_interval': ['N'],
            'INS_turn': ['N'],
            'INS_distance': ['N'],
            'HC_rate': ['HCs', 'Frames'],
            'run_speed': ['Frames'],
            'HC_angle': ['HCs']
        }
        column_names = {
            'INS_interval': ta,
            'INS_turn': lr,
            'INS_distance': ta,
            'HC_rate': ta,
            'run_speed': ta,
            'HC_angle': lr
        }

        boxplot_data = []
        modulation_data = []

        # init
        boxplot_black_near = []
        boxplot_white_near = []
        boxplot_black_far = []
        boxplot_white_far = []

        # for all trials
        pref_trials = [
            self.trial[index] for index in sorted(
                np.unique(self.trial, return_index=True)[1])
        ]
        trial_numbers = [
            self.trial_number[index] for index in sorted(
                np.unique(
                    self.trial_number,
                    return_index=True)[1])
        ]
        track_numbers = [
            self.track_number[index] for index in sorted(
                np.unique(self.track_number,
                          return_index=True)[1])
        ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        full_condition = self.full_condition
        # print trial_numbers
        # print pref_trials
        for trial_number in trial_numbers:

            # if len(pref_trials) != trial_number:
            #     dumpclean(pref_trials)
            #     print trial_number

            # mean_INS_distance,
            # mean_INS_interval, step_turning_angle
            for track_number in track_numbers:
                if variable_name in ['INS_distance',
                                     'INS_interval',
                                     'INS_turn']:

                    dist_idx_near = np.where(self.distance < distance)
                    dist_idx_far = np.where(self.distance >= distance)

                    trial_idx = np.where(self.trial_number == trial_number)

                    track_idx = np.where(self.track_number == track_number)

                    # index of all timepoints for the current trial and track
                    # near the odor
                    trial_track_near = np.intersect1d(dist_idx_near, trial_idx)
                    trial_track_near = np.intersect1d(trial_track_near,
                                                      track_idx)
                    # For each step in step_idx is it in the timepoints
                    # trial_track_near? (boolean index)
                    step_trial_track_near_bidx = np.in1d(self.step_idx,
                                                         trial_track_near)

                    # Steps followed by steps for which
                    # step_trial_track_near_bidx is true
                    wanted_steps_near_bidx = (step_trial_track_near_bidx &
                                              self.next_event_is_step)

                    # index of all timepoints for the current trial and track
                    # far the odor
                    trial_track_far = np.intersect1d(dist_idx_far, trial_idx)
                    trial_track_far = np.intersect1d(trial_track_far,
                                                     track_idx)
                    # For each step in step_idx is it in the timepoints
                    # trial_track_far? (boolean index)
                    step_trial_track_far_bidx = np.in1d(self.step_idx,
                                                        trial_track_far)

                    # Steps followed by steps for which
                    # step_trial_track_far_bidx is true
                    wanted_steps_far_bidx = (step_trial_track_far_bidx &
                                             self.next_event_is_step)

                    # Time points that are steps followed by steps, and are near
                    # and are in this trial and this track
                    t_idx_near = self.step_idx[wanted_steps_near_bidx]

                    # Time points that are steps followed by steps, and are far
                    # and are in this trial and this track
                    t_idx_far = self.step_idx[wanted_steps_far_bidx]

                    if (not any(wanted_steps_near_bidx)):
                        continue
                    if (not any(wanted_steps_far_bidx)):
                        continue

                    near_bearing_angle = self.bearing_angle[t_idx_near]

                    far_bearing_angle = self.bearing_angle[t_idx_far]

                    near_weights = getattr(self, variable_name)[
                        wanted_steps_near_bidx]

                    far_weights = getattr(self, variable_name)[
                        wanted_steps_far_bidx]

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
                        near_idx_ok = near_weights < 10  # smaller than 10 secs
                        far_idx_ok = far_weights < 10  # smaller than 10 secs
                        near_bearing_angle = near_bearing_angle[near_idx_ok]
                        far_bearing_angle = far_bearing_angle[far_idx_ok]
                        near_weights = near_weights[near_idx_ok]
                        far_weights = far_weights[far_idx_ok]

                # HC rate
                if variable_name == 'HC_rate':
                    idx_not_nan = ~np.isnan(
                        self.HC_initiation)
                    near_idx_not_nan = (
                            idx_not_nan & (self.distance < distance))
                    far_idx_not_nan = (
                            idx_not_nan & (self.distance >= distance))
                    idx_trial = self.trial_number == trial_number

                    near_idx_not_nan = near_idx_not_nan * idx_trial
                    far_idx_not_nan = far_idx_not_nan * idx_trial

                    idx_track = self.track_number == track_number
                    near_idx_not_nan = near_idx_not_nan & idx_track
                    far_idx_not_nan = far_idx_not_nan & idx_track

                    if (not any(near_idx_not_nan)):
                        continue
                    if (not any(far_idx_not_nan)):
                        continue

                    # print idx_not_nan.shape
                    near_bearing_angle = self.bearing_angle[
                        near_idx_not_nan]
                    far_bearing_angle = self.bearing_angle[far_idx_not_nan]
                    near_weights = self.HC_initiation[
                                       near_idx_not_nan] / float(par['dt'])
                    far_weights = self.HC_initiation[
                                      far_idx_not_nan] / float(par['dt'])

                    HC_dists_near = self.HC_initiation[near_idx_not_nan] * 1
                    np.put(HC_dists_near,
                           np.where(HC_dists_near == 1)[0],
                           np.insert(np.diff(np.where(HC_dists_near == 1)),
                                     0, 0)
                           )
                    HC_dists_far = self.HC_initiation[near_idx_not_nan] * 1
                    np.put(HC_dists_far,
                           np.where(HC_dists_far == 1)[0],
                           np.insert(np.diff(np.where(HC_dists_far == 1)),
                                     0, 0)
                           )

                # Run Speed
                if variable_name == 'run_speed':
                    idx_not_nan = ~np.isnan(self.midpoint_speed)
                    idx_non_hc = self.HC == 0
                    # Leave some distance before and after HC
                    idx_non_hc = np.invert(np.convolve(
                        np.invert(idx_non_hc),
                        (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                    idx_non_hc = idx_non_hc * idx_not_nan
                    near_idx_non_hc = (
                            idx_non_hc & (self.distance < distance))
                    far_idx_non_hc = (
                            idx_non_hc & (self.distance >= distance))
                    idx_trial = self.trial_number == trial_number
                    idx_track = self.track_number == track_number
                    near_idx_non_hc = near_idx_non_hc * idx_trial
                    near_idx_non_hc = near_idx_non_hc * idx_track
                    far_idx_non_hc = far_idx_non_hc * idx_trial
                    far_idx_non_hc = far_idx_non_hc * idx_track
                    near_bearing_angle = self.bearing_angle[
                        near_idx_non_hc]
                    far_bearing_angle = self.bearing_angle[
                        far_idx_non_hc]
                    # weights = self.dict[
                    #        self.full_condition].centroid_speed[
                    #     idx_non_hc]
                    near_weights = self.midpoint_speed[
                        near_idx_non_hc]
                    far_weights = self.midpoint_speed[
                        far_idx_non_hc]

                # HC angle
                if variable_name == 'HC_angle':
                    dist_idx_near = np.where(self.distance < distance)
                    dist_idx_far = np.where(self.distance >= distance)

                    trial_idx = np.where(self.trial_number == trial_number)

                    track_idx = np.where(self.track_number == track_number)

                    # index of all timepoints for the current trial and track
                    # near the odor
                    trial_track_near = np.intersect1d(dist_idx_near, trial_idx)
                    trial_track_near = np.intersect1d(trial_track_near,
                                                      track_idx)

                    wanted_HCs_near_bidx = np.array([], dtype=bool)
                    if (len(trial_track_near) != 0):
                        # For each HC in HC_start_idx is it in the timepoints
                        # trial_track_near? (boolean index)
                        HC_start_trial_track_near_bidx = np.in1d(
                            self.HC_start_idx,
                            trial_track_near)
                        HC_angle_large = np.abs(self.HC_angle) > np.deg2rad(
                            par['large_HC'])

                        wanted_HCs_near_bidx = (HC_start_trial_track_near_bidx &
                                                HC_angle_large)

                    # index of all timepoints for the current trial and track
                    # far the odor
                    trial_track_far = np.intersect1d(dist_idx_far, trial_idx)
                    trial_track_far = np.intersect1d(trial_track_far,
                                                     track_idx)
                    wanted_HCs_far_bidx = np.array([], dtype=bool)
                    if (len(trial_track_far) != 0):
                        # For each HC in HC_start_idx is it in the timepoints
                        # trial_track_far? (boolean index)
                        HC_start_trial_track_far_bidx = np.in1d(
                            self.HC_start_idx,
                            trial_track_far)
                        HC_angle_large = np.abs(self.HC_angle) > np.deg2rad(
                            par['large_HC'])

                        wanted_HCs_far_bidx = (HC_start_trial_track_far_bidx &
                                               HC_angle_large)

                    # Time points that are large enough HCs, and are near
                    # and are in this trial and this track
                    t_idx_near = self.HC_start_idx[wanted_HCs_near_bidx]

                    # Time points that are large enough HCs, and are far
                    # and are in this trial and this track
                    t_idx_far = self.HC_start_idx[wanted_HCs_far_bidx]

                    if (not any(wanted_HCs_near_bidx) and
                            not any(wanted_HCs_far_bidx)):
                        continue

                    near_bearing_angle = self.bearing_angle[t_idx_near]

                    far_bearing_angle = self.bearing_angle[t_idx_far]

                    near_weights = getattr(self, variable_name)[
                        wanted_HCs_near_bidx]

                    far_weights = getattr(self, variable_name)[
                        wanted_HCs_far_bidx]

                # apend boxplotdata
                if variable_name in ['INS_distance',
                                     'INS_interval']:
                    idx_black = np.abs(near_bearing_angle) < par['to_range']
                    idx_white = np.abs(near_bearing_angle) > par['away_range']

                    boxplot_black_near.append(np.mean(near_weights[idx_black]))
                    boxplot_white_near.append(np.mean(near_weights[idx_white]))
                    boxplot_data.append([full_condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][0],
                                         np.mean(near_weights[idx_black])])
                    boxplot_data.append([full_condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][1],
                                         np.mean(near_weights[idx_white])])

                if variable_name in ['HC_rate', 'run_speed']:
                    near_idx_black = (
                            np.abs(near_bearing_angle) < par['to_range'])
                    near_idx_white = (
                            np.abs(near_bearing_angle) > par['away_range'])
                    far_idx_black = (
                            np.abs(far_bearing_angle) < par['to_range'])
                    far_idx_white = (
                            np.abs(far_bearing_angle) > par['away_range'])

                    near_val_black = np.mean(near_weights[near_idx_black])
                    near_val_white = np.mean(near_weights[near_idx_white])
                    if variable_name in ['HC_rate']:
                        ncount_black = [np.count_nonzero(
                            near_weights[near_idx_black]),
                            np.sum(near_idx_black)]
                        ncount_white = [np.count_nonzero(
                            near_weights[near_idx_white]),
                            np.sum(near_idx_white)]
                        fcount_black = [np.count_nonzero(
                            far_weights[far_idx_black]),
                            np.sum(far_idx_black)]
                        fcount_white = [np.count_nonzero(
                            far_weights[far_idx_white]),
                            np.sum(far_idx_white)]
                        mncount = [np.count_nonzero(near_weights),
                                   len(near_bearing_angle)]
                        mfcount = [np.count_nonzero(far_weights),
                                   len(far_bearing_angle)]
                        variance_black_near = np.nanvar(HC_dists_near[np.where(
                            HC_dists_near[near_idx_black] > 0)])
                        variance_white_near = np.nanvar(HC_dists_near[np.where(
                            HC_dists_near[near_idx_white] > 0)])
                        variance_black_far = np.nanvar(HC_dists_far[np.where(
                            HC_dists_far[far_idx_black] > 0)])
                        variance_white_far = np.nanvar(HC_dists_far[np.where(
                            HC_dists_far[far_idx_white] > 0)])
                    else:
                        ncount_black = [np.sum(near_idx_black)]
                        ncount_white = [np.sum(near_idx_white)]
                        fcount_black = [np.sum(far_idx_black)]
                        fcount_white = [np.sum(far_idx_white)]
                        mncount = [len(near_bearing_angle)]
                        mfcount = [len(far_bearing_angle)]
                        variance_black_near = np.nanvar(
                            near_weights[near_idx_black])
                        variance_white_near = np.nanvar(
                            near_weights[near_idx_white])
                        variance_black_far = np.nanvar(
                            far_weights[far_idx_black])
                        variance_white_far = np.nanvar(
                            far_weights[far_idx_white])

                    if (
                            len(near_idx_black) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(near_val_black))
                    ):
                        boxplot_black_near.append(
                            np.mean(near_weights[near_idx_black]))
                        boxplot_data.append([
                                                full_condition,
                                                trial_dict[trial_number],
                                                track_number,
                                                full_condition + ' ' +
                                                column_names[variable_name][0],
                                                np.mean(near_weights[near_idx_black])] +
                                            ncount_black,
                                            variance_black_near
                                            )
                    if (
                            len(near_idx_white) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(near_val_white))
                    ):
                        boxplot_white_near.append(
                            np.mean(near_weights[near_idx_white]))

                        boxplot_data.append([
                                                full_condition,
                                                trial_dict[trial_number],
                                                track_number,
                                                full_condition + ' ' +
                                                column_names[variable_name][1],
                                                np.mean(near_weights[near_idx_white])] +
                                            ncount_white,
                                            variance_white_near
                                            )

                    far_val_black = np.mean(far_weights[far_idx_black])
                    far_val_white = np.mean(far_weights[far_idx_white])

                    if (
                            len(far_idx_black) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(far_val_black))
                    ):
                        boxplot_black_far.append(
                            np.mean(far_weights[far_idx_black]))
                        boxplot_data.append([
                                                full_condition,
                                                trial_dict[trial_number],
                                                track_number,
                                                full_condition + ' ' +
                                                column_names[variable_name][2],
                                                np.mean(far_weights[far_idx_black])] +
                                            fcount_black,
                                            variance_black_far
                                            )
                    if (
                            len(far_idx_white) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(far_val_white))
                    ):
                        boxplot_white_far.append(
                            np.mean(far_weights[far_idx_white]))

                        boxplot_data.append([
                                                full_condition,
                                                trial_dict[trial_number],
                                                track_number,
                                                full_condition + ' ' +
                                                column_names[variable_name][3],
                                                np.mean(far_weights[far_idx_white])] +
                                            fcount_white,
                                            variance_white_far
                                            )

                    if (
                            ((len(near_idx_black) + len(near_idx_white)) >=
                             par["ind_lim_" + variable_name]) and
                            (len(near_idx_black) > 0) and
                            (len(near_idx_white) > 0)
                    ):
                        if variable_name in ['HC_rate']:
                            nb_data = (
                                    np.mean(near_weights[near_idx_white]) -
                                    np.mean(near_weights[near_idx_black]))
                        else:
                            nb_data = (
                                    np.mean(near_weights[near_idx_black]) -
                                    np.mean(near_weights[near_idx_white]))
                        nt_data = (
                                np.sum(near_weights[near_idx_black]) /
                                len(near_weights[near_idx_black]) +
                                np.sum(near_weights[near_idx_white]) /
                                len(near_weights[near_idx_white])
                        )
                        if (
                                nt_data != 0 and (not np.isnan(nt_data)) and
                                (not np.isnan(nb_data))
                        ):
                            modulation_data.append([
                                                       full_condition,
                                                       trial_dict[trial_number],
                                                       track_number,
                                                       full_condition + ' ' + ' near',
                                                       nb_data / nt_data] +
                                                   mncount
                                                   )
                            if (
                                    len(far_idx_black) + len(far_idx_white) >=
                                    par["ind_lim_" + variable_name] and
                                    len(far_idx_black) > 0 and
                                    len(far_idx_white) > 0
                            ):
                                fb_data = (
                                        np.sum(far_weights[far_idx_white]) /
                                        len(far_weights[far_idx_white]) -
                                        np.sum(far_weights[far_idx_black]) /
                                        len(far_weights[far_idx_black]))
                                ft_data = (
                                        np.sum(far_weights[far_idx_black]) /
                                        len(far_weights[far_idx_black]) +
                                        np.sum(far_weights[far_idx_white]) /
                                        len(far_weights[far_idx_white])
                                )
                        if (
                                ft_data != 0 and (not np.isnan(fb_data)) and
                                (not np.isnan(ft_data))
                        ):
                            modulation_data.append([
                                                       full_condition,
                                                       trial_dict[trial_number],
                                                       track_number,
                                                       full_condition + ' ' + ' far',
                                                       fb_data / ft_data] +
                                                   mfcount
                                                   )

                if variable_name in ['INS_turn', 'HC_angle']:
                    near_idx_black = near_bearing_angle < 0.
                    near_idx_white = near_bearing_angle > 0.
                    far_idx_black = far_bearing_angle < 0.
                    far_idx_white = far_bearing_angle > 0.
                    near_val_black = np.mean(near_weights[near_idx_black])
                    near_val_white = np.mean(near_weights[near_idx_white])
                    variance_black_near = np.nanvar(
                        near_weights[near_idx_black])
                    variance_white_near = np.nanvar(
                        near_weights[near_idx_white])
                    variance_black_far = np.nanvar(far_weights[far_idx_black])
                    variance_white_far = np.nanvar(far_weights[far_idx_white])
                    if (
                            len(near_idx_black) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(near_val_black))
                    ):
                        boxplot_black_near.append(
                            np.mean(near_weights[near_idx_black]))
                        boxplot_data.append([
                            full_condition,
                            trial_dict[trial_number],
                            track_number,
                            full_condition + ' ' +
                            column_names[variable_name][0],
                            np.mean(near_weights[near_idx_black]),
                            np.count_nonzero(near_idx_black),
                            variance_black_near
                        ])
                    if (
                            len(near_idx_white) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(near_val_white))
                    ):
                        boxplot_white_near.append(
                            np.mean(near_weights[near_idx_white]))

                        boxplot_data.append([
                            full_condition,
                            trial_dict[trial_number],
                            track_number,
                            full_condition + ' ' +
                            column_names[variable_name][1],
                            np.mean(near_weights[near_idx_white]),
                            np.count_nonzero(near_idx_white),
                            variance_white_near
                        ])

                    far_val_black = np.mean(far_weights[far_idx_black])
                    far_val_white = np.mean(far_weights[far_idx_white])
                    if (
                            len(far_idx_black) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(far_val_black))
                    ):
                        boxplot_black_far.append(
                            np.mean(far_weights[far_idx_black]))
                        boxplot_data.append([
                            full_condition,
                            trial_dict[trial_number],
                            track_number,
                            full_condition + ' ' +
                            column_names[variable_name][2],
                            np.mean(far_weights[far_idx_black]),
                            np.count_nonzero(far_idx_black),
                            variance_black_far
                        ])

                    if (
                            len(far_idx_white) >= par["ind_lim_" + variable_name]
                            and
                            (~np.isnan(far_val_white))
                    ):
                        boxplot_white_far.append(
                            np.mean(far_weights[far_idx_white]))
                        boxplot_data.append([
                            full_condition,
                            trial_dict[trial_number],
                            track_number,
                            full_condition + ' ' +
                            column_names[variable_name][3],
                            np.mean(far_weights[far_idx_white]),
                            np.count_nonzero(far_idx_white),
                            variance_white_far
                        ])


        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = (["Group",
                           "Trial Name",
                           "Track Number",
                           '/'.join(column_names[variable_name]) + ' near/far',
                           str(variable_name)] +
                          samplesize_names[variable_name] +
                          ["Ind. Variance"])
            df.to_excel(
                self.excelWriter,
                sheet_name=str(variable_name) + '_bearing_box_d',
                index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    str(variable_name) + '_bearing_box_d'])
            if variable_name in ['HC_rate', 'run_speed']:
                print("============== Modulation ==========")
                print(len(modulation_data))
                print(modulation_data[0:10])
                mdf = pandas.DataFrame(modulation_data)
                mdf.sort_values(by=[0, 1], inplace=True)
                mdf.columns = (["Group",
                                "Trial Name",
                                "Track Name",
                                ' near/far',
                                variable_name + "_mod"] +
                               samplesize_names[variable_name])
                mdf.to_excel(self.excelWriter,
                             sheet_name=str(variable_name) +
                                        '_mod_to_bearing_box_d',
                             index=False)
                mf.fixXLColumns(
                    df,
                    self.excelWriter.sheets[
                        str(variable_name) +
                        '_mod_to_bearing_box_d'])



    def individuals_tracks_on_dish(self, par, conditions):
        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # figure settings
            fig = plt.figure(
                figsize=(
                    par['fig_width'],
                    par['fig_width']), dpi=100)
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

            plt.setp(ax1, xlim=(-
                                par['radius_dish'] -
                                5, par['radius_dish'] +
                                5), ylim=(-par['radius_dish'] -
                                          5, par['radius_dish'] +
                                          5), xticks=[], yticks=[])
            accpercent = round(float(cond.accepted_track_count) / float(cond.total_track_count), 1)

            naccpercent = 1 - accpercent

            for track in cond.accepted_tracks:
                not_nan_bidx = track["track_bidx"]
                ax1.plot(cond.spine4[not_nan_bidx][:, 0],
                         cond.spine4[not_nan_bidx][:, 1],
                         lw=1, ls='-', color='green', alpha=0.05)
            ax1.annotate(str(accpercent * 100) + '% (' + str(cond.accepted_track_count) + ')', xy=(20, -40))
            plt.savefig(par['figure_dir']
                        + '/' +
                        'accepted_tracks_on_dish' + '_' + str(condition))
            plt.close()

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

            plt.setp(ax1, xlim=(-
                                par['radius_dish'] -
                                5, par['radius_dish'] +
                                5), ylim=(-par['radius_dish'] -
                                          5, par['radius_dish'] +
                                          5), xticks=[], yticks=[])

            for track in cond.not_accepted_tracks:
                not_nan_bidx = track["track_bidx"]
                ax1.plot(cond.spine4[not_nan_bidx][:, 0],
                         cond.spine4[not_nan_bidx][:, 1],
                         lw=1, ls='-', color='red', alpha=0.05)
            ax1.annotate(str(naccpercent * 100) + '% (' + str(cond.total_track_count - cond.accepted_track_count) + ')',
                         xy=(20, -40))
            plt.savefig(par['figure_dir']
                        + '/' +
                        'not_accepted_tracks_on_dish' + '_' + str(condition))
            plt.close()

    def individuals_plot(self,
                         par,
                         conditions,
                         limit=8):

        def plot_histograms(cond, arrays, cols, xrange, nbins):
            nbins = 10
            nc = cols
            nr = int(len(arrays) / cols)
            fig = plotly_tools.make_subplots(rows=nr, cols=nc, print_grid=False,
                                             )
            for i in range(0, nr):
                for j in range(0, nc):
                    fig.append_trace(
                        go.Histogram(
                            x=arrays[i + j],
                            text=['({}, {})'.format(i + 1, j + 1)],
                            histnorm="percent",
                            marker=dict(
                                color='rgb(148, 11, 26)'
                            ),
                            xbins={'start': xrange[0],
                                   'end': xrange[1],
                                   'size': float((
                                                         xrange[1] - xrange[0]) / float(nbins))
                                   },
                        ),
                        row=i + 1, col=j + 1)

            fig['layout']['showlegend'] = False
            fig['layout'].update(height=2000)
            for i in range(1, len(arrays) + 1):
                fig['layout']['xaxis' + str(i)].update(
                    range=[xrange[0], xrange[1]])
                fig['layout']['yaxis' + str(i)].update(
                    range=[0, 100])
            return fig

        def plot_track(cond, track):
            # Axes -ylimit -> ylimit
            ylimit = 180
            # Annotations -ymax -> ymax
            ymax = 500
            fig = plotly_tools.make_subplots(
                rows=15,
                cols=1,
                subplot_titles=(
                    'Track',
                    'Distance',
                    'Heading Angle',
                    'Bearing Angle',
                    'Bending Angle',
                    'Inter-step Turn',
                    'Inter-step Distance',
                    'Inter-step Interval',
                    'Tail Speed Forward',
                    'Midpoint Speed',
                    'Distance PREF',
                    'INS Reorientation',
                    'HC Reorientation',
                    'Head Vector Angular Speed'
                ),
                specs=[
                    [{'r': 0.7, 'rowspan': 2}],
                    [None], [{}], [{}], [{}], [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}],
                    [{}]
                ],
                horizontal_spacing=0.05,
                vertical_spacing=0.05,
                print_grid=False,
            )
            # not_nan_bidx = ~np.isnan(cond.spine4[:, 1])
            # not_nan_bidx = not_nan_bidx * track["track_bidx"]
            not_nan_bidx = track["track_bidx"]
            length = np.nansum(track["track_bidx"])

            HC_start_idx = cond.HC_start_idx[track["track_HC_bidx"]]

            HC_end_idx = cond.HC_end_idx[track["track_HC_bidx"]]

            HC_angle = cond.HC_angle[track["track_HC_bidx"]]

            large_HC = 0
            subthreshold = False
            large_HC_bidx = angleComp(HC_angle, large_HC,
                                      subthreshold)

            heading_angle_at_start = cond.heading_angle[~np.isnan(cond.heading_angle)][
                HC_start_idx[large_HC_bidx]]
            heading_angle_at_end = cond.heading_angle[~np.isnan(cond.heading_angle)][
                HC_end_idx[large_HC_bidx]]

            weights = np.rad2deg(
                np.abs(heading_angle_at_start) -
                np.abs(heading_angle_at_end))

            INS_heading_angles = cond.heading_angle[~np.isnan(cond.heading_angle)][
                cond.step_idx[track["track_step_bidx"]]]

            INS_reorient = np.diff(np.abs(INS_heading_angles))

            print((len(INS_reorient)))
            print((len(cond.INS_distance[track["track_step_bidx"]])))

            start_label = go.Scatter(
                x=[cond.spine4[not_nan_bidx][:, 0][0] + 1.],
                y=[cond.spine4[not_nan_bidx][:, 1][0] + 1.],
                text=["Start"],
                mode='text',
                name="Start",
                showlegend=False,
            )
            end_label = go.Scatter(
                x=[cond.spine4[not_nan_bidx][:, 0][-1] + 1.],
                y=[cond.spine4[not_nan_bidx][:, 1][-1] + 1.],
                text=["End"],
                mode='text',
                name="End",
                showlegend=False
            )
            route = go.Scatter(
                x=cond.spine4[not_nan_bidx][:, 0],
                y=cond.spine4[not_nan_bidx][:, 1],
                mode='lines',
                name="Track",
                connectgaps=False,
            )
            HC = go.Scatter(
                x=cond.spine4[cond.HC_start_idx[track["track_HC_bidx"]]][:, 0],
                y=cond.spine4[cond.HC_start_idx[track["track_HC_bidx"]]][:, 1],
                mode='markers',
                name="HC"
            )
            distance = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=cond.distance[track["track_bidx"]],
                mode='lines',
                name="Distance",
                showlegend=False)

            headingAngle = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=np.rad2deg(cond.heading_angle[track["track_bidx"]]),
                mode='lines',
                name="Heading",
                showlegend=False)

            bearingAngle = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=np.rad2deg(cond.bearing_angle[track["track_bidx"]]),
                mode='lines',
                name="Bearing",
                showlegend=False)

            bendingAngle = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=np.rad2deg(cond.bending_angle[track["track_bidx"]]),
                mode='lines',
                name="Bending",
                showlegend=False)

            INSTurn = go.Scatter(
                x=cond.time[cond.step_idx[track["track_step_bidx"]]],
                y=np.rad2deg(cond.INS_turn[track["track_step_bidx"]]),
                mode='lines',
                name="Inter-step Turn",
                showlegend=False)

            INSTurnRAvg = go.Scatter(
                x=cond.time[cond.step_idx[track["track_step_bidx"]]],
                y=np.convolve(
                    np.rad2deg(cond.INS_turn[track["track_step_bidx"]]),
                    np.ones((10,)) / 10, mode="full"),
                mode='lines',
                name="Inter-step Turn Average",
                showlegend=False)

            INSDistance = go.Scatter(
                x=cond.time[cond.step_idx[track["track_step_bidx"]]],
                y=cond.INS_distance[track["track_step_bidx"]],
                mode='lines',
                name="Inter-step Distance",
                showlegend=False)

            INSInterval = go.Scatter(
                x=cond.time[cond.step_idx[track["track_step_bidx"]]],
                y=cond.INS_interval[track["track_step_bidx"]],
                mode='lines',
                name="Inter-step Interval",
                showlegend=False)

            INSReorient = go.Scatter(
                x=cond.time[cond.step_idx[track["track_step_bidx"]]],
                y=INS_reorient,
                mode='lines',
                name="Inter-step Distance",
                showlegend=False)

            tailSpeedFwd = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=cond.tail_speed_forward[track["track_bidx"]],
                mode='lines',
                name="Tail Speed fwd",
                showlegend=False)

            midpointSpeedFwd = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=cond.midpoint_speed[track["track_bidx"]],
                mode='lines',
                name="Midpoint Speed fwd",
                showlegend=False)

            distancePREF = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=cond.distance[track["track_bidx"]] / 115 * -1,
                mode='lines',
                name="Distance",
                showlegend=False)
            HC_reorientation = go.Scatter(
                x=cond.time[cond.HC_start_idx[track["track_HC_bidx"]]],
                y=weights,
                mode='markers',
                name="HC reorientation"
            )

            headVectorAngSpeed = go.Scatter(
                x=cond.time[track["track_bidx"]],
                y=cond.head_vector_angular_speed[track["track_bidx"]],
                mode='lines',
                name="Head Vector Angular",
                showlegend=False)

            fig.append_trace(start_label, 1, 1)
            fig.append_trace(end_label, 1, 1)
            fig.append_trace(route, 1, 1)
            fig.append_trace(HC, 1, 1)
            fig.append_trace(distance, 3, 1)
            fig.append_trace(headingAngle, 4, 1)
            fig.append_trace(bearingAngle, 5, 1)
            fig.append_trace(bendingAngle, 6, 1)
            fig.append_trace(INSTurn, 7, 1)
            fig.append_trace(INSTurnRAvg, 7, 1)
            fig.append_trace(INSDistance, 8, 1)
            fig.append_trace(INSInterval, 9, 1)
            fig.append_trace(tailSpeedFwd, 10, 1)
            fig.append_trace(midpointSpeedFwd, 11, 1)
            fig.append_trace(distancePREF, 12, 1)
            fig.append_trace(INSReorient, 13, 1)
            fig.append_trace(HC_reorientation, 14, 1)
            fig.append_trace(headVectorAngSpeed, 15, 1)

            border = 1.1

            fig['layout'].update(height=3500, width=2100)
            fig['layout'].update(
                legend=dict(
                    x=0.0,
                    y=1
                ))
            fig['layout'].update(title=(
                    "Track: " + str(int(track["track_number"])) +
                    " in trial: " + str(track["trial"]) + " PREF: " +
                    str(track["PREF"]) + " Frames: " + str(length)))

            fig['layout']['yaxis14'].update(
                range=[-np.pi / 2, np.pi / 2],
                showgrid=True,
                zeroline=True,
            )

            fig['layout']['xaxis14'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis13'].update(
                range=[-180, 180],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis13'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis12'].update(
                range=[-2, 2],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis12'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis11'].update(
                range=[-1, 0],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis11'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis10'].update(
                range=[0, 2.5],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis10'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis9'].update(
                range=[0, 2.5],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis9'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )

            fig['layout']['yaxis8'].update(
                range=[0, 2],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis8'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis7'].update(
                range=[0, 2],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis7'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis6'].update(
                range=[-ylimit / 5., ylimit / 5.],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis6'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis5'].update(
                range=[-ylimit, ylimit],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis5'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis4'].update(
                range=[-ylimit, ylimit],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis4'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis3'].update(
                range=[-ylimit, ylimit],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis3'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['yaxis2'].update(
                range=[0,
                       2 * par['radius_dish'] * border],
                showgrid=True,
                zeroline=True,
            )
            fig['layout']['xaxis2'].update(
                range=[par['start_time'], par['end_time']],
                showgrid=False,
                zeroline=True,
            )
            fig['layout']['xaxis1'].update(
                range=[-par['radius_dish'] * border,
                       par['radius_dish'] * border],
                showgrid=False,
                zeroline=False,
            )
            fig['layout']['yaxis1'].update(
                range=[-par['radius_dish'] * border,
                       par['radius_dish'] * border],
                zeroline=True
            )
            HC_shapes = []
            for idx, hc_start_idx in enumerate(
                    cond.HC_start_idx[track["track_HC_bidx"]]):
                if cond.HC_angle[idx] > 0:
                    color = 'rgba(200, 0, 0, 0.3)'
                else:
                    color = 'rgba(0, 200, 0, 0.3)'

                HC_shape = {
                    'type': 'rect',
                    'xref': 'x2',
                    'yref': 'y2',
                    'x0': cond.time[
                        cond.HC_start_idx[track["track_HC_bidx"]][idx]],
                    'y0': -ymax,
                    'x1': cond.time[
                        cond.HC_end_idx[track["track_HC_bidx"]][idx]],
                    'y1': ymax,
                    'line': {
                        'width': 0,
                    },
                    'fillcolor': color,
                }
                HC_shapes.append(copy.copy(HC_shape))
                for i in range(3, 15):
                    HC_shape['xref'] = 'x' + str(i)
                    HC_shape['yref'] = 'y' + str(i)
                    HC_shapes.append(copy.copy(HC_shape))

            for idx in cond.step_idx[track["track_step_bidx"]]:
                HC_shape = {
                    'type': 'line',
                    'xref': 'x2',
                    'yref': 'y2',
                    'x0': cond.time[idx],
                    'y0': -ymax,
                    'x1': cond.time[idx],
                    'y1': ymax,
                    'line': {
                        'width': 0.5,
                        'color': 'rgba(150, 150, 150, 0.5)',
                        'dash': 'dash',
                    },
                }
                HC_shapes.append(copy.copy(HC_shape))
                for i in range(3, 15):
                    HC_shape['xref'] = 'x' + str(i)
                    HC_shape['yref'] = 'y' + str(i)
                    HC_shapes.append(copy.copy(HC_shape))

            fig['layout'].update(shapes=[
                                            # petridish
                                            {
                                                'type': 'circle',
                                                'xref': 'x1',
                                                'yref': 'y1',
                                                'x0': -par['radius_dish'],
                                                'y0': -par['radius_dish'],
                                                'x1': par['radius_dish'],
                                                'y1': par['radius_dish'],
                                                'line': {
                                                    'color': 'rgba(50, 171, 96, 1)',
                                                },
                                            },
                                            # odour cup
                                            {
                                                'type': 'circle',
                                                'xref': 'x1',
                                                'yref': 'y1',
                                                'fillcolor': 'rgba(50, 171, 96, 0.7)',
                                                'x0': -4.0,
                                                'y0': cond.odor_A[track['trial_number']][1] - 4.,
                                                'y1': cond.odor_A[track['trial_number']][1] + 4.,
                                                'x1': 4.0,
                                                'line': {
                                                    'color': 'rgba(50, 171, 96, 1)',
                                                },
                                            },
                                        ] + HC_shapes
                                 )
            return fig

        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]

            n_tracks = len(cond.accepted_tracks)
            limit = np.min([limit, n_tracks])
            # track_sample = np.sort(
            #     np.random.permutation(np.arange(n_tracks))[:limit]).astype(int)
            # atracks = np.array(cond.accepted_tracks)
            random.seed(0)
            m = 12
            df_Abs_HC_Angles = []
            df_Abs_HC_Angles_to = []
            df_Abs_HC_Angles_away = []
            df_Abs_HC_Angles_b_to = []
            df_Abs_HC_Angles_b_away = []
            df_step_turns = []
            df_step_turns_to = []
            df_step_turns_away = []
            df_step_turns_b_to = []
            df_step_turns_b_away = []
            df_HC_rate = []
            df_HC_rate_b_to = []
            df_HC_rate_b_away = []
            """for track in random.sample(cond.accepted_tracks, m*limit):
                df_Abs_HC_Angles.append(
                    np.rad2deg(np.abs(cond.HC_angle[track["track_HC_bidx"]])))
                df_step_turns.append(
                    np.rad2deg(np.abs(cond.INS_turn[track["track_step_bidx"]])))
                df_HC_rate.append(np.diff(
                    cond.HC_start_idx[track["track_HC_bidx"]])/par['fps'])

            ploff.plot(plot_histograms(cond=cond,
                                       arrays=df_Abs_HC_Angles,
                                       cols=6,
                                       xrange=[0, 25],
                                       nbins=25),
                       output_type='file',
                       filename=par['figure_dir'] + "/individuals/" +
                       str(condition) + "_ABS_HC_Angles.html",
                       show_link=False,
                       validate=False,
                       auto_open=False
                       )

            ploff.plot(plot_histograms(cond=cond,
                                       arrays=df_step_turns,
                                       cols=6,
                                       xrange=[0, 25],
                                       nbins=25),
                       output_type='file',
                       filename=par['figure_dir'] + "/individuals/" +
                       str(condition) + "_step_turns.html",
                       show_link=False,
                       validate=False,
                       auto_open=False
                       )

            ploff.plot(plot_histograms(cond=cond,
                                       arrays=df_HC_rate,
                                       cols=6,
                                       xrange=[0, 50],
                                       nbins=25),
                       output_type='file',
                       filename=par['figure_dir'] + "/individuals/" +
                       str(condition) + "_HC_rate.html",
                       show_link=False,
                       validate=False,
                       auto_open=False
                       )"""
            print((len(cond.accepted_tracks)))
            for track in random.sample(cond.accepted_tracks, limit):
                ploff.plot(plot_track(cond, track),
                           output_type='file',
                           filename=par['figure_dir'] + "/individuals/" +
                                    str(condition) + '_Trial_' +
                                    str(track['trial']) +
                                    "_track_" + str(track["track_number"]) + "g_" +
                                    str(track['track_id']) + ".html",
                           # image_filename=par['figure_dir'] + "/individuals/" +
                           # str(condition) + '_Trial_' +
                           # str(track['trial']) +
                           # "_track_" + str(track["track_number"]) + "g_" +
                           # str(track['track_id']) + ".svg",
                           show_link=False,
                           validate=False,
                           auto_open=False,
                           # image='svg',
                           # image_width=2100,
                           # image_height=3500
                           )

    def create_individual_tables(self,
                                 par,
                                 conditions,
                                 var):
        # init
        frames = []
        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            for track in cond.accepted_tracks:
                if var == 'HC':
                    track_HC_bidx = track["track_HC_bidx"]

                    bearing_angle = cond.bearing_angle[
                        cond.HC_start_idx[track_HC_bidx]]
                    time = cond.time[
                        cond.HC_start_idx[track_HC_bidx]]
                    distance = cond.distance[
                        cond.HC_start_idx[track_HC_bidx]]
                    avg_distance = np.nanmean(distance)
                    if len(time) == 0:
                        continue

                    HC_start_idx = cond.HC_start_idx[track_HC_bidx]
                    HC_end_idx = cond.HC_end_idx[track_HC_bidx]

                    heading_angle_at_start = cond.heading_angle[HC_start_idx]
                    heading_angle_at_end = cond.heading_angle[HC_end_idx]

                    HC_Re = np.rad2deg(np.abs(heading_angle_at_start) -
                                       np.abs(heading_angle_at_end))
                    HC_Ac = np.rad2deg(np.abs(heading_angle_at_end))

                    # Bearing angle to 1 away -1
                    TurnTo = ((bearing_angle < par['to_range']) * 2.0) - 1.0
                    HC_Angle = np.rad2deg(cond.HC_angle[track_HC_bidx])
                    # Negative angle if away, Positive if towards odor
                    HC_angle_to_odor = HC_Angle * TurnTo
                    HC_Mag = np.fabs(HC_Angle)
                    HC_df = pandas.DataFrame(time)
                    HC_interval = np.diff(time)
                    HC_interval = np.insert(HC_interval, 0, np.nan)
                    HC_df.columns = ['time']
                    HC_df['condition'] = len(time) * [condition]
                    HC_df['trial'] = len(time) * [track['trial']]
                    HC_df['trial_track'] = len(time) * [track['track_number']]
                    HC_df['track_id'] = len(time) * [track['track_id']]
                    HC_df['bearing_angle'] = np.rad2deg(bearing_angle)
                    HC_df['distance'] = distance
                    HC_df['PREF'] = len(time) * [track['PREF']]
                    HC_df['HC_reorientation'] = HC_Re
                    HC_df['HC_accuracy'] = HC_Ac
                    HC_df['HC_angle'] = HC_Angle
                    HC_df['HC_magnitude'] = HC_Mag
                    HC_df['HC_relative_to_odor'] = HC_angle_to_odor
                    HC_df['HC_interval'] = HC_interval
                    HC_df['avg_distance'] = len(time) * [avg_distance]
                    frames.append(HC_df)
                if var == 'run_speed':
                    time = cond.time[0::16 * 2]
                    HC_df = pandas.DataFrame(time, columns=['time'])
                    HC_df['condition'] = len(time) * [condition]
                    HC_df['trial'] = len(time) * [track['trial']]
                    HC_df['trial_track'] = len(time) * [track['track_number']]
                    HC_df['track_id'] = len(time) * [track['track_id']]
                    HC_df['tail_speed'] = cond.tail_speed_forward[0::16 * 2]
                    frames.append(HC_df)

        Ind_df = pandas.concat(frames)
        Ind_df.to_excel(self.excelWriter,
                        sheet_name="Individual_HC_data",
                        index=False,
                        na_rep="NA")
        mf.fixXLColumns(
            Ind_df,
            self.excelWriter.sheets["Individual_HC_data"]
        )
        self.excel_titles["Individual_HC_data"] = "Individual HC data"

    def create_grid_exploration_graphs(self,
                                       par,
                                       conditions,
                                       var,
                                       grid_size=10):

        # Generate grid by create a grid:
        x = np.linspace(-par['radius_dish'], par['radius_dish'], grid_size + 1)
        y = np.linspace(-par['radius_dish'], par['radius_dish'], grid_size + 1)

        for condition_idx, condition in enumerate(conditions):
            cond = self.dict[condition]
            # for all accepted tracks
            for track in cond.accepted_tracks:
                dstr = description[variable_name]
                dstrshort = str(variable_name)
                not_nan_bidx = track["track_bidx"]
                length = np.nansum(track["track_bidx"])
                cond.spine4[not_nan_bidx][:, 0]
                cond.spine4[not_nan_bidx][:, 1]

    def fig_general_plot(self,
                         par,
                         conditions):
        for condition in conditions:
            cond = self.dict[condition]
            print(cond.tail_speed_forward.shape)
            print(cond.head_vector_angular_speed.shape)
            idx_not_nan = ~np.isnan(cond.tail_speed_forward)
            idx_lz = cond.tail_speed_forward > 0
            idx_not_nan = idx_lz * idx_not_nan
