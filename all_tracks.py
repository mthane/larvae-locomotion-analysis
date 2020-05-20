import pandas
import os
import numpy
from Tracks import Tracks
import copy
# from Plotlyplots import *
import numpy as np
from MiscFunctions import MiscFunctions as mf

class All_tracks:
    '''Simple dictionary of tracks from different conditions from one experiment,
        and respective figure functions'''

    def __init__(self, par, experiment, grouped_conditions):

        self.experiment = experiment
        self.dict = {}
        groups_name = '--'.join(
            str(p.replace('/', '_')) for p in par['groups'])
        if par['save_data']:
            self.excelWriter = pandas.ExcelWriter(
                "multiple"+
                '.xlsx', engine='xlsxwriter')
            print('init ExcelWriter')
        self.groups_name = groups_name
        self.plotly_included = False
        self.plotly_filename = par['figure_dir'] + "/"
        self.plotly_columns = []
        self.excel_titles = {}
        if par['save_data']:
            df = pandas.DataFrame()
            df.to_excel(self.excelWriter,
                        sheet_name='Table of Contents',
                        index=False)
        group_condition_array = [x.split('_-_') for x in grouped_conditions]
        # for all conditions
        # tmpdir = par['parent_dir'] + '/' + par['experiment_name']
        # + '/' + 'tmp'
        tmpdir = (par['parent_dir'] + '/' +
                  par['experiment_name'] + '/' +
                  par['tracked_on'] + '/' + 'tmp')
        for group_condition in group_condition_array:
            print("====" + str(group_condition))
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
                self.dict[group + '_-_' + condition] = load_pkl(
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

        # figure parameters
        # self.names = {
        #     'No_odor': 'No odor',
        #     'Naive_1in5000': 'Naive 1:5000',
        #     'Naive_1in500': 'Naive 1:500',
        #     'Naive_1in50': 'Naive 1:50',
        #     'Naive_1in5': 'Naive 1:5'}
        self.names = conditions
        self.names_short = self.names
        # self.names_short = {
        #     'No_odor': 'No odor',
        #     'Naive_1in5000': '1:5000',
        #     'Naive_1in500': '1:500',
        #     'Naive_1in50': '1:50',
        #     'Naive_1in5': '1:5'}

        colors = ['r', 'g', 'c', 'orange', 'blue', 'black', 'purple',
                  'pink', 'magenta', 'goldenrod', 'saddlebrown', 'grey']

        self.lc = dict(list(zip(conditions, colors)))

        linewidths = [1.0] * 12
        self.lw = dict(list(zip(conditions, linewidths)))
        alphas = [1.0] * 12
        self.alpha = dict(list(zip(conditions, alphas)))

#    def __new__(arg):
#        print("Calling __new__ with arg: " + arg)



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
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        # for all conditions
        # print conditions
        for condition in conditions:
            cur_cond = self.dict[condition]
            # mean_INS_distance, mean_INS_interval,
            # step_turning_angle
            if variable_name in ['INS_distance', 'INS_interval',
                                 'INS_turn']:

                bearing_angle = self.dict[condition].bearing_angle[
                    self.dict[condition].step_idx[
                        self.dict[condition].next_event_is_step]]

                weights = getattr(self.dict[condition], variable_name)[
                    self.dict[condition].next_event_is_step]

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
            if variable_name in ['HC_rate', 'HCs']:
                if subthreshold:
                    dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                # FIX FILTER HERE
                large_HC_idx = mf.angleComp(self.dict[condition].HC_angle,
                                         large_HC,
                                         subthreshold)

                idx_not_nan = ~np.isnan(self.dict[condition].HC_initiation)
                bearing_angle = self.dict[condition].bearing_angle[idx_not_nan]

                filtered_HC_rate = np.zeros(
                    self.dict[condition].HC_initiation.shape)
                filtered_HC_rate[
                    self.dict[condition].HC_start_idx[large_HC_idx]] = 1
                filtered_HC_rate = filtered_HC_rate[idx_not_nan]

                weights = filtered_HC_rate / float(par['dt'])

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(self.dict[condition].midpoint_speed)
                idx_non_hc = self.dict[condition].HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)
                idx_non_hc = idx_non_hc * idx_not_nan
                bearing_angle = self.dict[condition].bearing_angle[
                    idx_non_hc]
                # weights = self.dict[self.full_condition].centroid_speed[
                #     idx_non_hc]
                weights = self.dict[condition].midpoint_speed[
                    idx_non_hc]

            # HC angle
            if variable_name in ['HC_angle', 'Abs_HC_angle']:
                if subthreshold:
                    dstr = "(max angle " + str(large_HC) + ")"
                    dstrshort = "_maxA" + str(int(large_HC))
                else:
                    dstr = "(min angle " + str(large_HC) + ")"
                    dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(self.dict[condition].HC_angle,
                                         large_HC,
                                         subthreshold)
                bearing_angle = self.dict[condition].bearing_angle[
                    self.dict[condition].HC_start_idx[large_HC_idx]]

                if variable_name in ['Abs_HC_angle']:
                    weights = np.abs(np.rad2deg(
                        self.dict[condition].HC_angle[large_HC_idx]))
                else:
                    weights = np.rad2deg(
                        self.dict[condition].HC_angle[large_HC_idx])

            # Proportion of HC towards/away
            if variable_name in ['HC Proportion']:
                HC_angle = cur_cond.HC_angle
                large_HC_idx = angleComp(HC_angle,
                                         large_HC,
                                         subthreshold)
                HC_start_idx = cur_cond.HC_start_idx
                bearing_angle = self.dict[condition].bearing_angle[
                    HC_start_idx[large_HC_idx]]
                weights = np.rad2deg(
                    HC_angle[large_HC_idx])

                idx_towards = bearing_angle * weights < 0.
                idx_away = bearing_angle * weights > 0.
                weights = idx_towards.astype(int)

            # Proportion of Reorientation (positive/negative)
            if (variable_name == 'HC_Proportion_Reor'):
                HC_angle = cur_cond.HC_angle
                large_HC_idx = angleComp(HC_angle,
                                         large_HC,
                                         subthreshold)

                HC_start_idx = cur_cond.HC_start_idx
                HC_end_idx = cur_cond.HC_end_idx

                bearing_angle = self.dict[condition].bearing_angle[
                    HC_start_idx[large_HC_idx]]
                heading_angle_at_start = cur_cond.heading_angle[
                    HC_start_idx[large_HC_idx]]
                heading_angle_at_end = cur_cond.heading_angle[
                    HC_end_idx[large_HC_idx]]
                weights = np.rad2deg(
                    np.abs(heading_angle_at_start) -
                    np.abs(heading_angle_at_end))
                weights = np.array([x > 0 for x in weights]).astype(int)

            # add data for circular boundary conditions
            bearing_angle = np.hstack(
                [bearing_angle - 2 *
                 np.pi, bearing_angle, bearing_angle + 2 * np.pi])
            weights = np.tile(weights, 3)

            # hist
            n_samples = np.histogram(bearing_angle, bins=edges_bearing,
                                     normed=False)[0]

            hist = np.histogram(bearing_angle, bins=edges_bearing,
                                normed=False, weights=weights)[0]

            if (variable_name != 'HCs'): hist = hist / n_samples

            if (variable_name in ['HC Proportion', 'HC_Proportion_Reor']):
                hist = binned_statistic(bearing_angle, weights, bins=edges_bearing)[0]

            # convolve, filter width = 60 degree
            hist = np.convolve(np.ones(11) / 11., hist, mode='same')
            # plot
            line_data = np.vstack((line_data, hist))

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
            blist = createPlotlyLinePlot(
                df=df,
                xrange=data_xrange,
                title=title,
                description="Brief description",
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True

            self.plotly_columns = self.plotly_columns + blist
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[short_title]
            )
            self.excel_titles[short_title] = "LinePlot: " + title

    def figure_boxplot_variable_depending_on_bearing(
            self,
            par,
            conditions,
            variable_name,
            subthreshold=False,
            large_HC=-1,
            towards_away_limited='nl'):

        if (large_HC == -1):
            large_HC = par['large_HC']
        lsuffix = ""
        long_lsuffix = ''
        par['to_range'] = np.deg2rad(90)
        par['away_range'] = np.deg2rad(90)
        if (towards_away_limited == 'l'):
            lsuffix = '_l'
            long_lsuffix = 'limited'
        if (towards_away_limited == 'il'):
            lsuffix = '_il'
            long_lsuffix = 'inverse limited'
        # figure settings
        # ylabel = {
        #     'INS_interval': 'Inter-step-interval (s)',
        #     'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
        #     'INS_distance': 'Inter-step-distance (mm)',
        #     'HC_rate': 'HC rate (1/s)',
        #     'HC_angle': 'HC angle (' + degree_sign + ')',
        #     'Abs_HC_angle_turn_TA': 'HC angle (' + degree_sign + ')',
        #     'Abs_HC_angle_head_TA': 'HC angle (' + degree_sign + ')',
        #     'run_speed': 'Run Speed(mm/s)'
        #  }
        description = {
            'INS_interval': 'Inter-step-interval to bearing',
            'INS_turn': 'Inter-step-turn to bearing',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate to bearing',
            'HCs': 'HCs to bearing ' + long_lsuffix,
            'HC_angle': 'HC angle to bearing',
            'Abs_HC_angle_turn_TA': 'Absolute HC angle when turning' +
                                    ' to/away from the odour',
            'Abs_HC_angle_head_TA': 'Absolute HC angle when heading' +
                                    ' to/away from the odour',
            'run_speed': 'Run Speed to bearing'
        }

        column_names = {
            'INS_interval': ['toward', 'away'],
            'INS_turn': ['left', 'right'],
            'INS_distance': ['toward', 'away'],
            'HC_rate': ['toward', 'away'],
            'HCs': ['toward', 'away'],
            'run_speed': ['toward', 'away'],
            'HC_angle': ['left', 'right'],
            'Abs_HC_angle_turn_TA': ['turn towards', 'turn away'],
            'Abs_HC_angle_head_TA': ['toward', 'away']
        }
        boxplot_data = []
        modulation_data = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # full_condition = self.dict[condition].full_condition
            # print "========================"
            # print trial_numbers
            # print pref_trials
            # print "========================"
            for trial_number in trial_numbers:

                dstr = description[variable_name]
                dstrshort = str(variable_name)

                # if len(pref_trials) != trial_number:
                #     dumpclean(pref_trials)
                #     print trial_number

                # mean_INS_distance,
                # mean_INS_interval, step_turning_angle
                if variable_name in ['INS_distance',
                                     'INS_interval',
                                     'INS_turn']:

                    idx_trial = (
                            self.dict[condition].trial_number[
                                self.dict[condition].step_idx[
                                    self.dict[condition].next_event_is_step]] ==
                            trial_number)

                    bearing_angle = self.dict[condition].bearing_angle[
                        self.dict[condition].step_idx[
                            self.dict[condition].next_event_is_step]]
                    bearing_angle = bearing_angle[idx_trial]

                    weights = getattr(self.dict[condition], variable_name)[
                        self.dict[condition].next_event_is_step]
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
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    # FIX FILTER HERE
                    large_HC_idx = angleComp(self.dict[condition].HC_angle,
                                             large_HC,
                                             subthreshold)
                    idx_not_nan = ~np.isnan(self.dict[condition].HC_initiation)
                    idx_trial = self.dict[
                                    condition].trial_number == trial_number
                    bearing_angle = self.dict[condition].bearing_angle[
                        idx_not_nan * idx_trial]
                    filtered_HC_rate = np.zeros(
                        self.dict[condition].HC_initiation.shape)
                    filtered_HC_rate[
                        self.dict[condition].HC_start_idx[large_HC_idx]] = 1
                    filtered_HC_rate = filtered_HC_rate[idx_not_nan * idx_trial]
                    weights = filtered_HC_rate / float(par['dt'])

                # HCs
                if variable_name == 'HCs':
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    # FIX FILTER HERE
                    large_HC_idx = angleComp(self.dict[condition].HC_angle,
                                             large_HC,
                                             subthreshold)
                    idx_not_nan = ~np.isnan(self.dict[condition].HC_initiation)
                    idx_trial = self.dict[
                                    condition].trial_number == trial_number
                    bearing_angle = self.dict[condition].bearing_angle[
                        idx_not_nan * idx_trial]
                    filtered_HC_rate = np.zeros(
                        self.dict[condition].HC_initiation.shape)
                    filtered_HC_rate[
                        self.dict[condition].HC_start_idx[large_HC_idx]] = 1
                    filtered_HC_rate = filtered_HC_rate[idx_not_nan * idx_trial]
                    weights = filtered_HC_rate

                # Run Speed
                if variable_name == 'run_speed':
                    idx_not_nan = ~np.isnan(self.dict[condition].midpoint_speed)
                    idx_non_hc = self.dict[condition].HC == 0
                    # Leave some distance before and after HC
                    idx_non_hc = np.invert(np.convolve(
                        np.invert(idx_non_hc),
                        (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                    idx_non_hc = idx_non_hc * idx_not_nan
                    idx_trial = self.dict[
                                    condition].trial_number == trial_number
                    bearing_angle = self.dict[condition].bearing_angle[
                        idx_non_hc * idx_trial]
                    # weights = self.dict[self.full_condition].centroid_speed[
                    #     idx_non_hc * idx_trial]
                    weights = self.dict[condition].midpoint_speed[
                        idx_non_hc * idx_trial]

                # HC angle
                if variable_name in ['HC_angle', 'Abs_HC_angle_turn_TA',
                                     'Abs_HC_angle_head_TA']:
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    large_HC_idx = angleComp(self.dict[condition].HC_angle,
                                             large_HC,
                                             subthreshold)

                    bearing_angle = self.dict[condition].bearing_angle[
                        self.dict[condition].HC_start_idx[large_HC_idx]]

                    weights = np.rad2deg(
                        self.dict[condition].HC_angle[large_HC_idx])

                    idx_trial = (
                            self.dict[condition].trial_number
                            [self.dict[condition].HC_start_idx[large_HC_idx]] ==
                            trial_number)

                    bearing_angle = bearing_angle[idx_trial]
                    weights = weights[idx_trial]

                # apend boxplotdata
                if variable_name in ['INS_distance',
                                     'INS_interval']:
                    idx_black = np.abs(bearing_angle) < par['to_range']
                    idx_white = np.abs(bearing_angle) > par['away_range']

                    boxplot_data.append([condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][0],
                                         np.mean(weights[idx_black])])
                    boxplot_data.append([condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][1],
                                         np.mean(weights[idx_white])])

                if variable_name in ['HC_rate', 'run_speed', 'HCs']:

                    idx_black = np.abs(bearing_angle) < par['to_range']
                    idx_white = np.abs(bearing_angle) > par['away_range']
                    if (towards_away_limited == 'l'):
                        idx_black = np.abs(bearing_angle) < np.deg2rad(45)
                        idx_white = np.abs(bearing_angle) > np.deg2rad(135)
                    if (towards_away_limited == 'il'):
                        idx_black = np.array([x < np.deg2rad(90) and x > np.deg2rad(45)
                                              or x > np.deg2rad(90) and x < np.deg2rad(-45) for x in bearing_angle])
                        idx_white = np.array([x < np.deg2rad(135) and x > np.deg2rad(90)
                                              or x > np.deg2rad(-90) and x < np.deg2rad(-135) for x in bearing_angle])

                    if (variable_name != 'HCs'):
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            condition + ' ' + column_names[variable_name][0],
                            np.sum(weights[idx_black]) / len(weights[idx_black])
                        ])
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            condition + ' ' + column_names[variable_name][1],
                            np.sum(weights[idx_white]) / len(weights[idx_white])
                        ])
                    else:
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            condition + ' ' + column_names[variable_name][0],
                            np.sum(weights[idx_black])
                        ])
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            condition + ' ' + column_names[variable_name][1],
                            np.sum(weights[idx_white])
                        ])

                    if variable_name in ['HC_rate']:
                        b_data = (
                                np.mean(weights[idx_white]) -
                                np.mean(weights[idx_black]))
                    else:
                        b_data = (
                                np.mean(weights[idx_black]) -
                                np.mean(weights[idx_white]))
                    t_data = (
                            np.mean(weights[idx_black]) +
                            np.mean(weights[idx_white])
                    )

                    if variable_name == 'HCs':
                        b_data = (
                                np.sum(weights[idx_white]) -
                                np.sum(weights[idx_black]))

                        t_data = (
                                np.sum(weights[idx_black]) +
                                np.sum(weights[idx_white])
                        )

                    modulation_data.append([
                        condition,
                        trial_dict[trial_number],
                        b_data / t_data])

                if variable_name in ['INS_turn',
                                     'HC_angle',
                                     'Abs_HC_angle_turn_TA',
                                     'Abs_HC_angle_head_TA']:
                    if variable_name in ['Abs_HC_angle_turn_TA']:
                        # Turn Towards
                        idx_black = bearing_angle * weights < 0.
                        # Turn Away
                        idx_white = bearing_angle * weights > 0.
                        weights = np.abs(weights)
                    elif variable_name in ['Abs_HC_angle_head_TA']:
                        # Moving towards
                        idx_black = np.abs(bearing_angle) < par['to_range']
                        # Moving away
                        idx_white = np.abs(bearing_angle) > par['away_range']
                        weights = np.abs(weights)
                    else:
                        idx_black = bearing_angle < 0.
                        idx_white = bearing_angle > 0.

                    boxplot_data.append([
                        condition,
                        trial_dict[trial_number],
                        condition + ' ' + column_names[variable_name][0],
                        np.mean(weights[idx_black])
                    ])
                    boxplot_data.append([
                        condition,
                        trial_dict[trial_number],
                        condition + ' ' + column_names[variable_name][1],
                        np.mean(weights[idx_white])
                    ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Conditions",
                          str(variable_name)]

            blist = createPairedPlotlyBoxPlot(
                df=df,
                title=dstr,
                description="Brief description",
                values=str(variable_name),
                label="Group",
                pair_at="Conditions",
                pair_by=[column_names[variable_name][0],
                         column_names[variable_name][1]],
                do_stats=True,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True

            blist1 = createPlotlyBoxPlot(
                df=df[df['Conditions'].str.contains(
                    column_names[variable_name][0])],
                title=dstr + " (" +
                      column_names[variable_name][0] + ")",
                description="Brief description",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            blist2 = createPlotlyBoxPlot(
                df=df[df['Conditions'].str.contains(
                    column_names[variable_name][1])],
                title="Boxplot:  " + dstr + " (" +
                      column_names[variable_name][1] + ")",
                description="Brief description",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            self.plotly_columns = self.plotly_columns + blist
            self.plotly_columns = self.plotly_columns + blist1
            self.plotly_columns = self.plotly_columns + blist2
            title = dstr
            short_title = dstrshort + '_box' + lsuffix
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])

            self.excel_titles[short_title] = title

            if variable_name in ['HC_rate', 'run_speed', 'HCs']:
                mdf = pandas.DataFrame(modulation_data)
                mdf.sort_values(by=[0, 1], inplace=True)
                mdf.columns = ["Group",
                               "Trial Name",
                               str(variable_name) + '_modulation']
                title = (dstr +
                         " modulation depending on bearing " + long_lsuffix)
                short_title = dstrshort + '_mod_to_be_b' + lsuffix
                mblist = createPlotlyBoxPlot(
                    df=mdf,
                    title=title,
                    description=dstr + " modulation based on " +
                                str(column_names[variable_name]) + " " + long_lsuffix,
                    base=0.0,
                    values=str(variable_name) + '_modulation',
                    label="Group",
                    do_stats=True)

                self.plotly_columns = self.plotly_columns + mblist
                mdf.to_excel(self.excelWriter,
                             sheet_name=short_title,
                             header=["Group",
                                     "Trial Name",
                                     str(variable_name) + '_modulation'],
                             index=False)
                fixXLColumns(
                    mdf,
                    self.excelWriter.sheets[short_title])
                self.excel_titles[short_title] = title

    def figure_time_resolved_PREFs(
            self,
            par,
            conditions, ):

        # parameters
        # TODO arrange the limit better
        bin_edges = np.arange(par['start_time'], par['end_time'] + par['dt'],
                              par['dt'])

        # n_cond = len(conditions)

        # for all conditions
        # for condition in conditions:
        line_data = np.array(bin_edges)[:-1]
        print(len(line_data))
        for condition_idx, condition in enumerate(conditions):
            # print "------- SPINE4 " + condition + " -----------"
            # print self.dict[condition].spine4
            # print "------- END SPINE4 " + condition + " -----------"

            # init
            pref = []#np.array([])

            # for all trials
            for trial_number in np.unique(self.dict[condition].trial_number):
                # init
                time_up = 0.
                time_down = 0.

                # select trial and not nan idx
                trial_idx = self.dict[condition].trial_number == trial_number
                not_nan_idx = ~np.isnan(self.dict[condition].spine4[:, 0])

                # time and spine4_y
                time_tmp = self.dict[condition].time[not_nan_idx * trial_idx]
                spine4_y = self.dict[condition].spine4[
                    not_nan_idx *
                    trial_idx,
                    1]

                # time up and down
                time_down = np.histogram(time_tmp[spine4_y < 0.],
                                         bin_edges)[0]
                time_up = np.histogram(time_tmp[spine4_y > 0.],
                                       bin_edges)[0]
                time_sum = (time_down + time_up).astype(float)
                time_sum = time_sum.astype(np.float32, copy=False)
                time_sum[time_sum == 0] = np.nan
                cur_pref = ((
                                    time_up - time_down) /
                            time_sum.astype(float))
                
                pref.append(np.array(cur_pref))
                pref = np.vstack([pref, cur_pref]) if len(pref) else cur_pref

            line_data = np.vstack((line_data, np.nanmedian(pref,axis=0)))
            line_data = np.vstack([line_data, np.nanpercentile(pref,25,axis=0)])
            line_data = np.vstack([line_data, np.nanpercentile(pref,75,axis=0)])


        # save data
        if par['save_data']:
            percentiles = ['', ' 25%', ' 75%']
            column_names = [a +
                            b for a in conditions for b in percentiles]
            column_names.insert(0, 'Time (s)')
            df = pandas.DataFrame(line_data.T)
            df.columns = column_names

            df.to_excel(self.excelWriter,
                        sheet_name='time_resolved_PREF',
                        header=column_names,
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'time_resolved_PREF'])
            self.excel_titles[
                'time_resolved_PREF'] = "LinePlot: Time resolved PREF"

    def figure_boxplot_variable_distance_split(
            self,
            par,
            conditions,
            variable_name,
            distance,
            subthreshold=False,
            large_HC=-1):

        # this function takes very long to compute, because of ...== trial

        if (large_HC == -1):
            large_HC = par['large_HC']

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
        #     'run_speed': 'Run Speed(mm/s)',
        #     'Abs_HC_angle': 'HC angle (' + degree_sign + ')',
        # }

        description = {
            'INS_interval': 'Inter-step-interval to bearing',
            'INS_turn': 'Inter-step-turn to bearing',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate',
            'HC_angle': 'HC angle',
            'Abs_HC_angle': 'Absolute HC angle',
            'run_speed': 'Run Speed'
        }

        boxplot_data = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            full_condition = self.dict[condition].full_condition
            # print trial_numbers
            # print pref_trials
            cur_cond = copy.copy(self.dict[condition])
            for trial_number in trial_numbers:
                dstr = description[variable_name]
                dstrshort = str(variable_name)
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
                    idx_trial = cur_cond.trial_number == trial_number
                    near_idx_non_hc = near_idx_non_hc * idx_trial
                    far_idx_non_hc = far_idx_non_hc * idx_trial
                    near_weights = cur_cond.midpoint_speed[
                        near_idx_non_hc]
                    far_weights = cur_cond.midpoint_speed[
                        far_idx_non_hc]
                if variable_name in ['HC_rate']:
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    # FIX FILTER HERE
                    large_HC_idx = angleComp(self.dict[condition].HC_angle,
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
                    idx_trial = cur_cond.trial_number == trial_number
                    near_idx_not_nan = near_idx_not_nan * idx_trial
                    far_idx_not_nan = far_idx_not_nan * idx_trial

                    filtered_HC_rate = np.zeros(
                        self.dict[condition].HC_initiation.shape)
                    filtered_HC_rate[
                        self.dict[condition].HC_start_idx[large_HC_idx]] = 1

                    near_weights = filtered_HC_rate[
                                       near_idx_not_nan] / float(par['dt'])
                    far_weights = filtered_HC_rate[
                                      far_idx_not_nan] / float(par['dt'])
                # HC angle
                if variable_name in ['HC_angle',
                                     'Abs_HC_angle']:
                    near_HC_start_idx = cur_cond.HC_start_idx[
                        cur_cond.distance[cur_cond.HC_start_idx] < distance]
                    far_HC_start_idx = cur_cond.HC_start_idx[
                        cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                    near_HC_angle = cur_cond.HC_angle[
                        cur_cond.distance[cur_cond.HC_start_idx] < distance]
                    far_HC_angle = cur_cond.HC_angle[
                        cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    near_large_HC_idx = angleComp(
                        near_HC_angle,
                        large_HC,
                        subthreshold)
                    far_large_HC_idx = angleComp(
                        far_HC_angle,
                        large_HC,
                        subthreshold)

                    near_idx_trial = (
                            cur_cond.trial_number
                            [near_HC_start_idx[near_large_HC_idx]] ==
                            trial_number)
                    far_idx_trial = (
                            cur_cond.trial_number
                            [far_HC_start_idx[far_large_HC_idx]] ==
                            trial_number)

                    near_weights = np.rad2deg(
                        near_HC_angle[near_large_HC_idx])
                    far_weights = np.rad2deg(
                        far_HC_angle[far_large_HC_idx])

                    if variable_name in ['Abs_HC_angle']:
                        near_weights = np.abs(near_weights[near_idx_trial])
                        far_weights = np.abs(far_weights[far_idx_trial])
                    else:
                        near_weights = near_weights[near_idx_trial]
                        far_weights = far_weights[far_idx_trial]

                boxplot_data.append([
                    condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + ' near',
                    np.sum(near_weights) / len(
                        near_weights)
                ])
                boxplot_data.append([
                    condition,
                    trial_dict[trial_number],
                    full_condition + ' ' + ' far',
                    np.sum(far_weights) / len(
                        far_weights)
                ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)

            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          'Near/Far',
                          str(variable_name)]
            title = dstr + " split distance"
            short_title = dstrshort + '_b_d'
            blist_far = createPlotlyBoxPlot(
                df=df[df['Near/Far'].str.contains('far')],
                title="Boxplot " + dstr + " far",
                description="Distance > " + str(distance) + 'mm',
                values=str(variable_name),
                label="Group",
                do_stats=True,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True

            blist_near = createPlotlyBoxPlot(
                df=df[df['Near/Far'].str.contains('near')],
                title="Boxplot " + dstr + " near",
                description="Distance < " + str(distance) + 'mm',
                values=str(variable_name),
                label="Group",
                do_stats=True)

            self.plotly_columns = self.plotly_columns + blist_far
            self.plotly_columns = self.plotly_columns + blist_near
            df.to_excel(
                self.excelWriter,
                sheet_name=short_title,
                index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])
            self.excel_titles[short_title] = title

    def figure_boxplot_variable_depending_on_bearing_distance_split(
            self,
            par,
            conditions,
            variable_name,
            distance,
            subthreshold=False,
            large_HC=-1,
            towards_away_limited='nl'
    ):

        # this function takes very long to compute, because of ...== trial
        lsuffix = ""
        long_lsuffix = ""
        if (towards_away_limited == 'l'):
            lsuffix = '_l'
            long_lsuffix = 'limited'
        if (towards_away_limited == 'il'):
            lsuffix = '_il'
            long_lsuffix = 'inverse limited'

        if (large_HC == -1):
            large_HC = par['large_HC']

        # figure settings
        # ylabel = {
        #     'INS_interval': 'Inter-step-interval (s)',
        #     'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
        #     'INS_distance': 'Inter-step-distance (mm)',
        #     'HC_rate': 'HC rate (1/s)',
        #     'HC_angle': 'HC angle (' + degree_sign + ')',
        #     'run_speed': 'Run Speed(mm/s)',
        #     'Abs_HC_angle_turn_TA': 'HC angle (' + degree_sign + ')',
        #     'Abs_HC_angle_head_TA': 'HC angle (' + degree_sign + ')',
        # }
        description = {
            'INS_interval': 'Inter-step-interval to bearing',
            'INS_turn': 'Inter-step-turn to bearing',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate to bearing',
            'HC_angle': 'HC angle to bearing',
            'HCs': 'HCs to bearing',
            'Abs_HC_angle_turn_TA': 'Absolute HC angle when turning' +
                                    ' to/away from the odour',
            'Abs_HC_angle_head_TA': 'Absolute HC angle when heading' +
                                    ' to/away from the odour',
            'run_speed': 'Run Speed to bearing'
        }

        column_names = {
            'INS_interval': ['toward', 'away'],
            'INS_turn': ['left', 'right'],
            'INS_distance': ['toward', 'away'],
            'HC_rate': ['toward', 'away'],
            'HCs': ['toward', 'away'],
            'run_speed': ['toward', 'away'],
            'HC_angle': ['left', 'right'],
            'Abs_HC_angle_turn_TA': ['turn towards', 'turn away'],
            'Abs_HC_angle_head_TA': ['toward', 'away'],
        }

        boxplot_data = []
        modulation_data = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            full_condition = self.dict[condition].full_condition
            # print trial_numbers
            # print pref_trials
            cur_cond = copy.copy(self.dict[condition])
            for trial_number in trial_numbers:

                dstr = description[variable_name]
                dstrshort = str(variable_name)
                # if len(pref_trials) != trial_number:
                #     dumpclean(pref_trials)
                #     print trial_number

                # mean_INS_distance,
                # mean_INS_interval, step_turning_angle
                if variable_name in ['INS_distance',
                                     'INS_interval',
                                     'INS_turn']:

                    near_next_event_step_idx = cur_cond.next_event_is_step[
                        cur_cond.distance[cur_cond.step_idx[
                            cur_cond.next_event_is_step]] < distance
                        ]
                    far_next_event_step_idx = cur_cond.next_event_is_step[
                        cur_cond.distance[cur_cond.step_idx[
                            cur_cond.next_event_is_step]] >= distance
                        ]

                    near_idx_trial = (
                            cur_cond.trial_number[
                                cur_cond.step_idx[
                                    near_next_event_step_idx]] == trial_number)

                    far_idx_trial = (
                            cur_cond.trial_number[
                                cur_cond.step_idx[
                                    far_next_event_step_idx]] == trial_number)

                    near_bearing_angle = cur_cond.bearing_angle[
                        cur_cond.step_idx[
                            near_next_event_step_idx]]
                    near_bearing_angle = near_bearing_angle[near_idx_trial]

                    far_bearing_angle = cur_cond.bearing_angle[
                        cur_cond.step_idx[
                            far_next_event_step_idx]]
                    far_bearing_angle = far_bearing_angle[far_idx_trial]

                    near_weights = getattr(cur_cond, variable_name)[
                        near_next_event_step_idx]
                    near_weights = near_weights[near_idx_trial]

                    far_weights = getattr(cur_cond, variable_name)[
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
                        near_idx_ok = near_weights < 10  # smaller than 10 secs
                        far_idx_ok = far_weights < 10  # smaller than 10 secs
                        near_bearing_angle = near_bearing_angle[near_idx_ok]
                        far_bearing_angle = far_bearing_angle[far_idx_ok]
                        near_weights = near_weights[near_idx_ok]
                        far_weights = far_weights[far_idx_ok]

                # HC rate
                if variable_name in ['HC_rate', 'HCs']:
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    # FIX FILTER HERE
                    large_HC_idx = angleComp(self.dict[condition].HC_angle,
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
                    idx_trial = cur_cond.trial_number == trial_number
                    near_idx_not_nan = near_idx_not_nan * idx_trial
                    far_idx_not_nan = far_idx_not_nan * idx_trial

                    filtered_HC_rate = np.zeros(
                        self.dict[condition].HC_initiation.shape)
                    filtered_HC_rate[
                        self.dict[condition].HC_start_idx[large_HC_idx]] = 1

                    near_bearing_angle = cur_cond.bearing_angle[
                        near_idx_not_nan]
                    far_bearing_angle = cur_cond.bearing_angle[
                        far_idx_not_nan]
                    if (variable_name == 'HC_rate'):
                        near_weights = filtered_HC_rate[
                                           near_idx_not_nan] / float(par['dt'])
                        far_weights = filtered_HC_rate[
                                          far_idx_not_nan] / float(par['dt'])
                    else:
                        near_weights = filtered_HC_rate[
                            near_idx_not_nan]
                        far_weights = filtered_HC_rate[
                            far_idx_not_nan]
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
                    idx_trial = cur_cond.trial_number == trial_number
                    near_idx_non_hc = near_idx_non_hc * idx_trial
                    far_idx_non_hc = far_idx_non_hc * idx_trial
                    near_bearing_angle = cur_cond.bearing_angle[
                        near_idx_non_hc]
                    far_bearing_angle = cur_cond.bearing_angle[
                        far_idx_non_hc]
                    # weights = cur_cond.dict[
                    #        cur_cond.full_condition].centroid_speed[
                    #     idx_non_hc]
                    near_weights = cur_cond.midpoint_speed[
                        near_idx_non_hc]
                    far_weights = cur_cond.midpoint_speed[
                        far_idx_non_hc]

                # HC angle
                if variable_name in ['HC_angle',
                                     'Abs_HC_angle_turn_TA',
                                     'Abs_HC_angle_head_TA']:
                    near_HC_start_idx = cur_cond.HC_start_idx[
                        cur_cond.distance[cur_cond.HC_start_idx] < distance]
                    far_HC_start_idx = cur_cond.HC_start_idx[
                        cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                    near_HC_angle = cur_cond.HC_angle[
                        cur_cond.distance[cur_cond.HC_start_idx] < distance]
                    far_HC_angle = cur_cond.HC_angle[
                        cur_cond.distance[cur_cond.HC_start_idx] >= distance]

                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ") " + long_lsuffix
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ") " + long_lsuffix
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    near_large_HC_idx = angleComp(
                        near_HC_angle,
                        large_HC,
                        subthreshold)
                    far_large_HC_idx = angleComp(
                        far_HC_angle,
                        large_HC,
                        subthreshold)

                    near_idx_trial = (
                            cur_cond.trial_number
                            [near_HC_start_idx[near_large_HC_idx]] ==
                            trial_number)
                    far_idx_trial = (
                            cur_cond.trial_number
                            [far_HC_start_idx[far_large_HC_idx]] ==
                            trial_number)

                    near_bearing_angle = cur_cond.bearing_angle[
                        near_HC_start_idx[near_large_HC_idx]]
                    far_bearing_angle = cur_cond.bearing_angle[
                        far_HC_start_idx[far_large_HC_idx]]

                    near_weights = np.rad2deg(
                        near_HC_angle[near_large_HC_idx])
                    far_weights = np.rad2deg(
                        far_HC_angle[far_large_HC_idx])

                    near_bearing_angle = near_bearing_angle[near_idx_trial]
                    far_bearing_angle = far_bearing_angle[far_idx_trial]

                    near_weights = near_weights[near_idx_trial]
                    far_weights = far_weights[far_idx_trial]

                # apend boxplotdata
                if variable_name in ['INS_distance',
                                     'INS_interval']:
                    idx_black = np.abs(near_bearing_angle) < par['to_range']
                    idx_white = np.abs(near_bearing_angle) > par['away_range']

                    boxplot_data.append([condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][0],
                                         np.mean(near_weights[idx_black])])
                    boxplot_data.append([condition,
                                         trial_dict[trial_number],
                                         column_names[variable_name][1],
                                         np.mean(near_weights[idx_white])])

                if variable_name in ['HC_rate', 'run_speed', 'HCs']:

                    near_idx_black = np.abs(near_bearing_angle) < par['to_range']
                    near_idx_white = np.abs(near_bearing_angle) > par['away_range']
                    far_idx_black = np.abs(far_bearing_angle) < par['to_range']
                    far_idx_white = np.abs(far_bearing_angle) > par['away_range']
                    if (towards_away_limited == 'l'):
                        near_idx_black = np.abs(near_bearing_angle) < np.deg2rad(45)
                        near_idx_white = np.abs(near_bearing_angle) > np.deg2rad(135)
                        far_idx_black = np.abs(far_bearing_angle) < np.deg2rad(45)
                        far_idx_white = np.abs(far_bearing_angle) > np.deg2rad(135)
                    if (towards_away_limited == 'il'):
                        near_idx_black = np.array([x < np.deg2rad(90) and x > np.deg2rad(45)
                                                   or x > np.deg2rad(90) and x < np.deg2rad(-45) for x in
                                                   near_bearing_angle])
                        near_idx_white = np.array([x < np.deg2rad(135) and x > np.deg2rad(90)
                                                   or x > np.deg2rad(-90) and x < np.deg2rad(-135) for x in
                                                   near_bearing_angle])
                        far_idx_black = np.array([x < np.deg2rad(90) and x > np.deg2rad(45)
                                                  or x > np.deg2rad(90) and x < np.deg2rad(-45) for x in
                                                  far_bearing_angle])
                        far_idx_white = np.array([x < np.deg2rad(135) and x > np.deg2rad(90)
                                                  or x > np.deg2rad(-90) and x < np.deg2rad(-135) for x in
                                                  far_bearing_angle])

                    if (variable_name == 'HCs'):
                        if (len(near_weights[near_idx_black]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][0] + ' near',
                                np.sum(near_weights[near_idx_black])
                            ])

                        if (len(near_weights[near_idx_white]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                str(column_names[variable_name][1]) + ' near',
                                np.sum(near_weights[near_idx_white])
                            ])

                        if (len(far_weights[far_idx_black]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][0] + ' far',
                                np.sum(far_weights[far_idx_black])
                            ])

                        if (len(far_weights[far_idx_white]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][1] + ' far',
                                np.sum(far_weights[far_idx_white])
                            ])

                        if (
                                (len(near_weights[near_idx_white]) > 0) and
                                (len(near_weights[near_idx_black]) > 0) and
                                (len(far_weights[far_idx_white]) > 0) and
                                (len(far_weights[far_idx_black]) > 0)):
                            nb_data = (
                                    np.sum(near_weights[near_idx_white]) -
                                    np.sum(near_weights[near_idx_black]))
                            fb_data = (
                                    np.sum(far_weights[far_idx_white]) -
                                    np.sum(far_weights[far_idx_black]))

                            nt_data = (
                                    np.sum(near_weights[near_idx_black]) +
                                    np.sum(near_weights[near_idx_white])
                            )
                            ft_data = (
                                    np.sum(far_weights[far_idx_black]) +
                                    np.sum(far_weights[far_idx_white])
                            )
                            modulation_data.append([
                                condition,
                                trial_dict[trial_number],
                                nb_data / nt_data,
                                fb_data / ft_data])
                    else:

                        if (len(near_weights[near_idx_black]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][0] + ' near',
                                np.sum(near_weights[near_idx_black]) / len(
                                    near_weights[near_idx_black])
                            ])

                        if (len(near_weights[near_idx_white]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                str(column_names[variable_name][1]) + ' near',
                                np.sum(near_weights[near_idx_white]) / len(
                                    near_weights[near_idx_white])
                            ])

                        if (len(far_weights[far_idx_black]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][0] + ' far',
                                np.sum(far_weights[far_idx_black]) / len(
                                    far_weights[far_idx_black])
                            ])

                        if (len(far_weights[far_idx_white]) > 0):
                            boxplot_data.append([
                                condition,
                                trial_dict[trial_number],
                                full_condition + ' ' +
                                column_names[variable_name][1] + ' far',
                                np.sum(far_weights[far_idx_white]) / len(
                                    far_weights[far_idx_white])
                            ])

                        # if variable_name in ['HC_rate']:
                        if (
                                (len(near_weights[near_idx_white]) > 0) and
                                (len(near_weights[near_idx_black]) > 0) and
                                (len(far_weights[far_idx_white]) > 0) and
                                (len(far_weights[far_idx_black]) > 0)):
                            if variable_name in ['HC_rate']:
                                nb_data = (
                                        np.mean(near_weights[near_idx_white]) -
                                        np.mean(near_weights[near_idx_black]))
                                fb_data = (
                                        np.mean(far_weights[far_idx_white]) -
                                        np.mean(far_weights[far_idx_black]))
                            else:
                                nb_data = (
                                        np.mean(near_weights[near_idx_black]) -
                                        np.mean(near_weights[near_idx_white]))
                                fb_data = (
                                        np.mean(far_weights[far_idx_black]) -
                                        np.mean(far_weights[far_idx_white]))
                            nt_data = (
                                    np.mean(near_weights[near_idx_black]) +
                                    np.mean(near_weights[near_idx_white])
                            )
                            ft_data = (
                                    np.mean(far_weights[far_idx_black]) +
                                    np.mean(far_weights[far_idx_white])
                            )
                            modulation_data.append([
                                condition,
                                trial_dict[trial_number],
                                nb_data / nt_data,
                                fb_data / ft_data])

                if variable_name in ['INS_turn', 'HC_angle',
                                     'Abs_HC_angle_turn_TA',
                                     'Abs_HC_angle_head_TA']:
                    if variable_name in ['Abs_HC_angle_turn_TA']:
                        # Turn Towards
                        near_idx_black = near_bearing_angle * near_weights < 0.
                        far_idx_black = far_bearing_angle * far_weights < 0.
                        # Turn Away
                        near_idx_white = near_bearing_angle * near_weights > 0.
                        far_idx_white = far_bearing_angle * far_weights > 0.
                        near_weights = np.abs(near_weights)
                        far_weights = np.abs(far_weights)
                    elif variable_name in ['Abs_HC_angle_head_TA']:
                        # Moving towards
                        near_idx_black = (
                                np.abs(near_bearing_angle) < par['to_range'])
                        far_idx_black = (
                                np.abs(far_bearing_angle) < par['to_range'])
                        # Moving away
                        near_idx_white = (
                                np.abs(near_bearing_angle) > par['away_range'])
                        far_idx_white = (
                                np.abs(far_bearing_angle) > par['away_range'])

                        near_weights = np.abs(near_weights)
                        far_weights = np.abs(far_weights)
                    else:
                        near_idx_black = near_bearing_angle < 0.
                        near_idx_white = near_bearing_angle > 0.
                        far_idx_black = far_bearing_angle < 0.
                        far_idx_white = far_bearing_angle > 0.

                    if (len(near_weights[near_idx_black]) > 0):
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            full_condition + ' ' +
                            column_names[variable_name][0] + ' near',
                            np.mean(near_weights[near_idx_black])
                        ])

                    if (len(near_weights[near_idx_white]) > 0):
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            full_condition + ' ' +
                            column_names[variable_name][1] + ' near',
                            np.mean(near_weights[near_idx_white])
                        ])

                    if (len(far_weights[far_idx_black]) > 0):
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            full_condition + ' ' +
                            column_names[variable_name][0] + ' far',
                            np.mean(far_weights[far_idx_black])
                        ])

                    if (len(far_weights[far_idx_white]) > 0):
                        boxplot_data.append([
                            condition,
                            trial_dict[trial_number],
                            full_condition + ' ' +
                            column_names[variable_name][1] + ' far',
                            np.mean(far_weights[far_idx_white])
                        ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          'Conditions',
                          str(variable_name)]

            df_near = df[df['Conditions'].str.contains('near')]
            df_far = df[df['Conditions'].str.contains('far')]

            blist_near = createPairedPlotlyBoxPlot(
                df=df_near,
                title=dstr + " (near)",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Group",
                pair_at="Conditions",
                pair_by=[column_names[variable_name][0] + " near",
                         column_names[variable_name][1] + " near"],
                do_stats=True,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True

            blist_far = createPairedPlotlyBoxPlot(
                df=df_far,
                title=dstr + " (far)",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Group",
                pair_at="Conditions",
                pair_by=[column_names[variable_name][0] + " far",
                         column_names[variable_name][1] + " far"],
                do_stats=True)

            blist_near_0 = createPlotlyBoxPlot(
                df=df_near[df_near['Conditions'].str.contains(
                    column_names[variable_name][0])],
                title=dstr + " (near/" +
                      column_names[variable_name][0] + ")",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            blist_near_1 = createPlotlyBoxPlot(
                df=df_near[df_near['Conditions'].str.contains(
                    column_names[variable_name][1])],
                title=dstr + " (near/" +
                      column_names[variable_name][1] + ")",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            blist_far_0 = createPlotlyBoxPlot(
                df=df_far[df_far['Conditions'].str.contains(
                    column_names[variable_name][0])],
                title=dstr + " (far/" +
                      column_names[variable_name][0] + ")",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            blist_far_1 = createPlotlyBoxPlot(
                df=df_far[df_far['Conditions'].str.contains(
                    column_names[variable_name][1])],
                title=dstr + " (far/" +
                      column_names[variable_name][1] + ")",
                description="Distance < " + str(distance) + "mm",
                values=str(variable_name),
                label="Conditions",
                do_stats=True)

            self.plotly_columns = self.plotly_columns + blist_near
            self.plotly_columns = self.plotly_columns + blist_near_0
            self.plotly_columns = self.plotly_columns + blist_near_1
            self.plotly_columns = self.plotly_columns + blist_far
            self.plotly_columns = self.plotly_columns + blist_far_0
            self.plotly_columns = self.plotly_columns + blist_far_1

            title = dstr + " split distance"
            short_title = dstrshort + 'b_d'

            df.to_excel(
                self.excelWriter,
                sheet_name=short_title,
                index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])

            self.excel_titles[short_title] = title

            if variable_name in ['HC_rate', 'run_speed', 'HCs']:
                mdf = pandas.DataFrame(modulation_data)
                mdf.sort_values(by=[0, 1], inplace=True)
                mdf.columns = ["Group",
                               "Trial Name",
                               'near', 'far']
                mblist_near = createPlotlyBoxPlot(
                    df=mdf,
                    title=dstr +
                          " modulation (near) " + long_lsuffix,
                    description=dstr + " modulation based on " +
                                str(column_names[variable_name]) + long_lsuffix,
                    values="near",
                    base=0.0,
                    label="Group",
                    do_stats=True)
                self.plotly_columns = self.plotly_columns + mblist_near
                self.excel_titles[short_title] = title

                mblist_far = createPlotlyBoxPlot(
                    df=mdf,
                    title=dstr +
                          " modulation (far)",
                    description=str(variable_name) + " modulation based on " +
                                str(column_names[variable_name]) + long_lsuffix,
                    base=0.0,
                    values="far",
                    label="Group",
                    do_stats=True)
                self.plotly_columns = self.plotly_columns + mblist_far
                title = (dstr +
                         " modulation , split distance " + long_lsuffix)
                short_title = dstrshort + '_mod_be_b_d' + lsuffix
                mdf.to_excel(self.excelWriter,
                             sheet_name=short_title,
                             index=False)
                fixXLColumns(
                    df,
                    self.excelWriter.sheets[short_title])
                self.excel_titles[short_title] = title

    def figure_proportion_of_HCs_boxplot(
            self,
            par,
            conditions,
            subthreshold=False,
            large_HC=-1):

        # this function takes very long to compute, because of ...== trial

        if (large_HC == -1):
            large_HC = par['large_HC']

        # ylabel = "Proportion of HCs towards"
        # column_names = ['To odor', 'Away from odor']
        # ylim = (0.0, 1.0)
        # yticks = np.arange(0.0, 1.1, 0.25)

        dstr = "Boxplot: Proportion of HCs towards the odour"
        dstrshort = "HC_proportion_box"

        if subthreshold:
            dstr = dstr + "(max angle " + str(large_HC) + ")"
            dstrshort = dstrshort + "_maxA" + str(int(large_HC))
        else:
            dstr = dstr + "(min angle " + str(large_HC) + ")"
            dstrshort = dstrshort + "_minA" + str(int(large_HC))

        boxplot_data = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # full_condition = self.dict[condition].full_condition
            for trial_number in trial_numbers:

                # if len(pref_trials) <= trial_number:
                #     dumpclean(pref_trials)
                #     print trial_number

                large_HC_idx = angleComp(
                    self.dict[condition].HC_angle,
                    large_HC,
                    subthreshold)

                bearing_angle = self.dict[condition].bearing_angle[
                    self.dict[condition].HC_start_idx[large_HC_idx]]
                idx_trial = (
                        self.dict[condition].trial_number
                        [self.dict[condition].HC_start_idx[large_HC_idx]] ==
                        trial_number)
                bearing_angle = bearing_angle[idx_trial]

                weights = np.rad2deg(
                    self.dict[condition].HC_angle[large_HC_idx])

                weights = weights[idx_trial]
                idx_towards = np.sum(bearing_angle * weights < 0.)
                idx_away = np.sum(bearing_angle * weights > 0.)
                if ((idx_away + idx_towards) == 0):
                    continue
                proportion_to = (
                        float(idx_towards) / float(idx_away + idx_towards))

                boxplot_data.append([
                    condition,
                    trial_dict[trial_number],
                    proportion_to
                    # proportion_away
                ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Towards"]
            blist = createPlotlyBoxPlot(
                df=df,
                title=dstr,
                description="Brief description",
                base=0.5,
                values="Towards",
                label="Group",
                do_stats=True,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True
            self.plotly_columns = self.plotly_columns + blist
            df.to_excel(self.excelWriter,
                        sheet_name=str(dstrshort),
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    dstrshort])
            self.excel_titles[dstrshort] = dstr

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
                    variable_name)[
                    near_next_event_step_idx]
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
                large_HC_idx = angleComp(self.dict[condition].HC_angle,
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

                near_large_HC_idx = angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = angleComp(far_HC_angle,
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

                near_large_HC_idx = angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = angleComp(far_HC_angle,
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

                near_large_HC_idx = angleComp(near_HC_angle,
                                              large_HC,
                                              subthreshold)
                far_large_HC_idx = angleComp(far_HC_angle,
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
            blist_near = createPlotlyLinePlot(
                df=df.ix[:, [0] + list(range(1, len(cur_column_names), 2))],
                xrange=data_xrange,
                title=dstr + "(near)",
                description="Brief description",
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True
            self.plotly_columns = self.plotly_columns + blist_near
            blist_far = createPlotlyLinePlot(
                df=df.ix[:, [0] + list(range(2, len(cur_column_names), 2))],
                xrange=data_xrange,
                title=dstr + "(far)",
                description="Brief description")
            self.plotly_columns = self.plotly_columns + blist_far
            title = dstr + " split distance"
            short_title = dstrshort + '_bear_line_d'
            df.to_excel(
                self.excelWriter,
                sheet_name=dstrshort + '_bear_line_d',
                index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    dstrshort + '_bear_line_d'])
            self.excel_titles[short_title] = "LinePlot: " + title

    def figure_proportion_of_HCs_boxplot_distance_split(
            self,
            par,
            conditions,
            distance,
            subthreshold=False,
            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        dstr = "Boxplot: Proportion of HCs towards the odour"
        dstrshort = "HC_proportion_box_d"

        if subthreshold:
            dstr = dstr + "(max angle " + str(large_HC) + ")"
            dstrshort = dstrshort + "_maxA" + str(int(large_HC))
        else:
            dstr = dstr + "(min angle " + str(large_HC) + ")"
            dstrshort = dstrshort + "_minA" + str(int(large_HC))

        boxplot_data = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # init
            boxplot_black = []
            boxplot_white = []

            near_HC_start_idx = self.dict[condition].HC_start_idx[
                self.dict[condition].distance[
                    self.dict[condition].HC_start_idx] < distance]
            far_HC_start_idx = self.dict[condition].HC_start_idx[
                self.dict[condition].distance[
                    self.dict[condition].HC_start_idx] >= distance]

            near_HC_angle = self.dict[condition].HC_angle[
                self.dict[condition].distance[
                    self.dict[condition].HC_start_idx] < distance]
            far_HC_angle = self.dict[condition].HC_angle[
                self.dict[condition].distance[
                    self.dict[condition].HC_start_idx] >= distance]

            near_large_HC_idx = angleComp(
                near_HC_angle,
                large_HC,
                subthreshold)
            far_large_HC_idx = angleComp(
                far_HC_angle,
                large_HC,
                subthreshold)

            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # full_condition = self.dict[condition].full_condition
            for trial_number in trial_numbers:

                # if len(pref_trials) <= trial_number:
                #     dumpclean(pref_trials)
                #     print trial_number

                near_idx_trial = (
                        self.dict[condition].trial_number
                        [near_HC_start_idx[near_large_HC_idx]] ==
                        trial_number)
                far_idx_trial = (
                        self.dict[condition].trial_number
                        [far_HC_start_idx[far_large_HC_idx]] ==
                        trial_number)

                near_bearing_angle = self.dict[condition].bearing_angle[
                    near_HC_start_idx[near_large_HC_idx]]
                far_bearing_angle = self.dict[condition].bearing_angle[
                    far_HC_start_idx[far_large_HC_idx]]

                near_weights = np.rad2deg(
                    near_HC_angle[near_large_HC_idx])
                far_weights = np.rad2deg(
                    far_HC_angle[far_large_HC_idx])

                near_bearing_angle = near_bearing_angle[near_idx_trial]
                far_bearing_angle = far_bearing_angle[far_idx_trial]

                near_weights = near_weights[near_idx_trial]
                far_weights = far_weights[far_idx_trial]

                idx_towards_near = np.sum(
                    near_bearing_angle * near_weights < 0.)
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
                # boxplot_white.append(proportion_away)
                boxplot_data.append([
                    condition,
                    trial_dict[trial_number],
                    condition + ' near',
                    proportion_towards_near
                ])
                boxplot_data.append([
                    condition,
                    trial_dict[trial_number],
                    condition + ' far',
                    proportion_towards_far
                ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          "Near/Far",
                          "value"]
            blist_far = createPlotlyBoxPlot(
                df=df[df['Near/Far'].str.contains('far')],
                title=dstr + "(far)",
                description="Distance > " + str(distance) + 'mm',
                base=0.5,
                values='value',
                label="Group",
                do_stats=True,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True

            blist_near = createPlotlyBoxPlot(
                df=df[df['Near/Far'].str.contains('near')],
                title=dstr + "(near)",
                description="Distance < " + str(distance) + 'mm',
                base=0.5,
                values='value',
                label="Group",
                do_stats=True)

            self.plotly_columns = self.plotly_columns + blist_far
            self.plotly_columns = self.plotly_columns + blist_near
            df.to_excel(self.excelWriter,
                        sheet_name=str(dstrshort),
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    dstrshort])
            self.excel_titles[dstrshort] = dstr + ' split distance'

    def figure_boxplot_variable(
            self,
            par,
            conditions,
            variable_name,
            subthreshold=False,
            large_HC=-1,
            quadrant=''):

        if (large_HC == -1):
            large_HC = par['large_HC']

        if variable_name in ["Abs_HC_angle_turn_TA"]:
            return
        if variable_name in ["Abs_HC_angle_head_TA"]:
            variable_name = "Abs_HC_angle"

        description = {
            'INS_interval': 'Inter-step-interval',
            'INS_turn': 'Inter-step-turn',
            'INS_distance': 'Inter-step-distance (mm)',
            'HC_rate': 'HC rate',
            'HC_angle': 'HC angle',
            'Abs_HC_angle': 'Absolute HC angle',
            'run_speed': 'Run Speed'
        }

        dstr = description[variable_name]
        dstrshort = str(variable_name)

        boxplot_data = []
        # for all conditions


        for condition_idx, condition in enumerate(conditions):


            # for all trials
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # full_condition = self.dict[condition].full_condition
            # print "========================"
            # print trial_numbers
            # print pref_trials
            # print "========================"
            for trial_number in trial_numbers:

                # if len(pref_trials) != trial_number:
                #     dumpclean(pref_trials)
                #     print trial_number
                # checking quadrants
                idx_trial = self.dict[
                                condition].trial_number == trial_number
                idx_quadrant =idx_trial
                trial_data = self.dict[condition]

                if (quadrant == 'q1'):
                    idx_quadrant = np.array([p[0] <= 0 and p[1] <= 0 for p in trial_data.spine4])
                if (quadrant == 'q2'):
                    idx_quadrant = np.array([p[0] > 0 and p[1] <= 0 for p in trial_data.spine4])
                if (quadrant == 'q3'):
                    idx_quadrant = np.array([p[0] <= 0 and p[1] > 0 for p in trial_data.spine4])
                if (quadrant == 'q4'):
                    idx_quadrant = np.array([p[0] > 0 and p[1] > 0 for p in trial_data.spine4])


                dstr = description[variable_name]
                dstrshort = str(variable_name)+' '+quadrant

                # mean_INS_distance,
                # mean_INS_interval, step_turning_angle
                if variable_name in ['INS_distance',
                                     'INS_interval',
                                     'INS_turn']:

                    idx_trial = (
                            self.dict[condition].trial_number[
                                self.dict[condition].step_idx[
                                    self.dict[condition].next_event_is_step]] ==
                            trial_number)

                    bearing_angle = self.dict[condition].bearing_angle[
                        self.dict[condition].step_idx[
                            self.dict[condition].next_event_is_step]]
                    bearing_angle = bearing_angle[idx_trial]

                    weights = getattr(self.dict[condition], variable_name)[
                        self.dict[condition].next_event_is_step]
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

                    idx_not_nan = ~np.isnan(self.dict[condition].HC_initiation)
                    idx_trial = self.dict[
                                    condition].trial_number == trial_number

                    bearing_angle = self.dict[condition].bearing_angle[
                        idx_not_nan * idx_trial]

                    filtered_HC_rate = np.zeros(
                        self.dict[condition].HC_initiation.shape)
                    filtered_HC_rate[
                        self.dict[condition].HC_start_idx[large_HC_idx]] = 1

                    filtered_HC_rate = filtered_HC_rate[idx_not_nan * idx_trial]

                    #filtering points in quadrants
                    if(quadrant!=''):
                        idx_quadrant = idx_quadrant[idx_trial * idx_not_nan]
                        filtered_HC_rate = filtered_HC_rate[idx_quadrant]
                    weights = filtered_HC_rate / float(par['dt'])

                # Run Speed
                if variable_name == 'run_speed':
                    idx_not_nan = ~np.isnan(self.dict[condition].midpoint_speed)
                    idx_non_hc = self.dict[condition].HC == 0
                    # Leave some distance before and after HC
                    idx_non_hc = np.invert(np.convolve(
                        np.invert(idx_non_hc),
                        (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                    idx_non_hc = idx_non_hc * idx_not_nan
                    idx_trial = self.dict[
                                    condition].trial_number == trial_number

                    bearing_angle = self.dict[condition].bearing_angle[
                        idx_non_hc * idx_trial]
                    # weights = self.dict[self.full_condition].centroid_speed[
                    #     idx_non_hc * idx_trial]
                    weights = self.dict[condition].midpoint_speed[
                        idx_non_hc * idx_trial]

                    #filtering points in quadrants

                    if (quadrant != ''):
                        idx_quadrant = idx_quadrant[idx_trial * idx_non_hc]
                        weights = weights[idx_quadrant]

                # HC angle
                if variable_name in ['HC_angle', 'Abs_HC_angle']:
                    if subthreshold:
                        dstr = dstr + "(max angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_maxA" + str(int(large_HC))
                    else:
                        dstr = dstr + "(min angle " + str(large_HC) + ")"
                        dstrshort = dstrshort + "_minA" + str(int(large_HC))

                    large_HC_idx = mf.angleComp(self.dict[condition].HC_angle,
                                             large_HC,
                                             subthreshold)

                    bearing_angle = self.dict[condition].bearing_angle[
                        self.dict[condition].HC_start_idx[large_HC_idx]]
                    weights = np.rad2deg(
                        self.dict[condition].HC_angle[large_HC_idx])

                    idx_trial = (
                            self.dict[condition].trial_number
                            [self.dict[condition].HC_start_idx[large_HC_idx]] ==
                            trial_number)
                    
                
                         
                    if variable_name in ['Abs_HC_angle']:
                        bearing_angle = bearing_angle[idx_trial]
                        weights = np.abs(weights[idx_trial])

                        # filtering points in quadrants
                        if(quadrant != ''):
                            idx_quadrant = idx_quadrant[self.dict[condition].HC_start_idx[large_HC_idx]]
                            idx_quadrant = idx_quadrant[idx_trial]
                            weights = weights[idx_quadrant]
                    else:
                        bearing_angle = bearing_angle[idx_trial]
                        weights = weights[idx_trial]

                # apend boxplotdata
                if variable_name in ['INS_distance',
                                     'INS_interval']:
                    boxplot_data.append([condition,
                                         trial_dict[trial_number],
                                         np.mean(weights)])

                if variable_name in ['HC_rate', 'run_speed']:
                    boxplot_data.append([
                        condition,
                        trial_dict[trial_number],
                        np.sum(weights) / len(weights)
                    ])

                if variable_name in ['INS_turn',
                                     'HC_angle',
                                     'Abs_HC_angle']:
                    boxplot_data.append([
                        condition,
                        trial_dict[trial_number],
                        np.mean(weights)
                    ])

        # save data
        if par['save_data']:
            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          str(variable_name)]
            title = dstr
            short_title = dstrshort + '_boxplot'
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])
            self.excel_titles[short_title] = title

    def figure_heatmap_distance_and_bearing(self, par, conditions,
                                            variable_name,
                                            subthreshold=False,
                                            large_HC=-1):

        if (large_HC == -1):
            large_HC = par['large_HC']

        if variable_name in ["Abs_HC_angle_turn_TA"]:
            return
        if variable_name in ["Abs_HC_angle_head_TA"]:
            variable_name = "Abs_HC_angle"

        # box filter
        box_half_points = 11
        box = np.ones((box_half_points * 2, box_half_points * 2))

        dstr = ""
        # dstrshort = ""

        dataframe_list = []
        zmin = None
        zmax = None
        for condition in conditions:
            cur_cond = copy.copy(self.dict[condition])
            # mean_INS_distance, mean_INS_interval, step_turning_angle
            if variable_name in ['INS_distance', 'INS_interval',
                                 'INS_turn']:
                distance = cur_cond.distance[cur_cond.step_idx[
                    cur_cond.next_event_is_step]]
                bearing_angle = cur_cond.bearing_angle[
                    cur_cond.step_idx[
                        cur_cond.next_event_is_step]]
                weights = getattr(self, variable_name)[
                    cur_cond.next_event_is_step]

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
                if subthreshold:
                    dstr = "(max angle " + str(large_HC) + ")"
                    # dstrshort = "_maxA" + str(int(large_HC))
                else:
                    dstr = "(min angle " + str(large_HC) + ")"
                    # dstrshort = "_minA" + str(int(large_HC))
                # FIX FILTER HERE
                large_HC_idx = angleComp(self.dict[condition].HC_angle,
                                         large_HC,
                                         subthreshold)

                idx_not_nan = ~np.isnan(cur_cond.HC_initiation)
                distance = cur_cond.distance[idx_not_nan]
                bearing_angle = cur_cond.bearing_angle[idx_not_nan]
                filtered_HC_rate = np.zeros(
                    self.dict[condition].HC_initiation.shape)
                filtered_HC_rate[
                    self.dict[condition].HC_start_idx[large_HC_idx]] = 1
                filtered_HC_rate = filtered_HC_rate[idx_not_nan]
                weights = filtered_HC_rate / float(par['dt'])

            # HC angle
            if variable_name in ['HC_angle']:
                if subthreshold:
                    dstr = "(max angle " + str(large_HC) + ")"
                    # dstrshort = "_maxA" + str(int(large_HC))
                else:
                    dstr = "(min angle " + str(large_HC) + ")"
                    # dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(cur_cond.HC_angle, large_HC,
                                         subthreshold)
                distance = (
                    cur_cond.distance[cur_cond.HC_start_idx[large_HC_idx]])
                bearing_angle = (
                    cur_cond.bearing_angle[cur_cond.HC_start_idx[large_HC_idx]])
                weights = np.rad2deg(cur_cond.HC_angle[large_HC_idx])

            # Absolute HC angle
            if variable_name in ['Abs_HC_angle']:
                if subthreshold:
                    dstr = "(max angle " + str(large_HC) + ")"
                    # dstrshort = "_maxA" + str(int(large_HC))
                else:
                    dstr = "(min angle " + str(large_HC) + ")"
                    # dstrshort = "_minA" + str(int(large_HC))
                large_HC_idx = angleComp(cur_cond.HC_angle, large_HC,
                                         subthreshold)
                distance = (
                    cur_cond.distance[cur_cond.HC_start_idx[large_HC_idx]])
                bearing_angle = (
                    cur_cond.bearing_angle[cur_cond.HC_start_idx[large_HC_idx]])
                weights = np.abs(np.rad2deg(cur_cond.HC_angle[large_HC_idx]))

            # Run Speed
            if variable_name == 'run_speed':
                idx_not_nan = ~np.isnan(cur_cond.midpoint_speed)
                idx_non_hc = cur_cond.HC == 0
                # Leave some distance before and after HC
                idx_non_hc = np.invert(np.convolve(
                    np.invert(idx_non_hc),
                    (par['gap'] * 2 + 1) * [1], mode='same') > 0)

                idx_non_hc = idx_non_hc * idx_not_nan
                distance = cur_cond.distance[idx_non_hc]
                bearing_angle = cur_cond.bearing_angle[idx_non_hc]
                weights = cur_cond.midpoint_speed[
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

            # color_bar_title = {
            #     'INS_interval': 'Inter-step-interval (s)',
            #     'INS_turn': 'Inter-step-turn (' + degree_sign + ')',
            #     'INS_distance': 'Inter-step-distance (mm)',
            #     'HC_rate': 'HC rate (1/s)',
            #     'HC_angle': 'HC angle (' + degree_sign + ')',
            #     'Abs_HC_angle': 'Absolute HC angle (' + degree_sign + ')',
            #     'run_speed': 'Run Speed(mm/s)'
            # }
            dist = par['edges_distance'][:-1]
            bear = np.rad2deg(par['edges_bearing'][:-1])
            nheatmap = np.vstack((bear, heatmap.T))
            nheatmap = np.vstack((dist, nheatmap))
            nheatmap = nheatmap.T
            df = pandas.DataFrame(nheatmap)
            dataframe_list.append(df)
            if zmin is None or np.min(np.min(df.ix[:, 2:])) < zmin:
                zmin = np.min(np.min(df.ix[:, 2:]))
                if zmax is None or np.max(np.max(df.ix[:, 2:])) > zmax:
                    zmax = np.max(np.max(df.ix[:, 2:]))

        crange = None
        if zmin is not None and zmax is not None:
            crange = [zmin, zmax]

        ind = 0
        for df in dataframe_list:
            # save data
            if par['save_data']:
                blist = createPlotlyHeatMap(
                    df=df,
                    yrange=[df.ix[:, 0].tolist()[box_half_points - 2],
                            df.ix[:, 0].tolist()[-box_half_points - 2]],
                    crange=crange,
                    title=str(variable_name) +
                          " wrt bearing and distance for " + conditions[ind] +
                          dstr,
                    description="Brief Description",
                    include_plotly=(not self.plotly_included)
                )
            if not self.plotly_included:
                self.plotly_included = True

            self.plotly_columns = self.plotly_columns + blist
            ind = ind + 1

    def figure_hist(self, par, conditions, key):

        if key in ["Abs_HC_angle_turn_TA"]:
            return
        if key in ["Abs_HC_angle_head_TA"]:
            key = "Abs_HC_angle"

        nbins = 100
        df = pandas.DataFrame()
        # total = np.array([])
        for condition in conditions:
            # default parameters
            x_false = None
            # vline = None
            cur_cond = copy.copy(self.dict[condition])
            if key == 'duration':
                x = cur_cond.duration
                x_true = x
                xname = 'Track duration'
                xlabel = "Track duration"
                xfname = None
                x_false = None
                xlabel = 'Duration (s)'
                xlim = (0,
                        cur_cond.par['end_time'] - cur_cond.par['start_time'])

            if key == 'tail_speed_forward':
                x = cur_cond.tail_speed_forward
                x_true = cur_cond.tail_speed_forward[cur_cond.step_idx]
                x_false = cur_cond.tail_speed_forward[cur_cond.step_idx_false]
                xlabel = 'Tail speed (mm/s)'
                xlim = (0, 3)

            if key == 'HC_angle':
                xname = 'Abs(HC angle) >= ' + str(par['large_HC'])
                xfname = 'Abs(HC angle) < ' + str(par['large_HC'])
                x = np.rad2deg(cur_cond.HC_angle)
                x_false = x[np.abs(x) < par['large_HC']]
                x_true = x[np.abs(x) >= par['large_HC']]
                xlabel = 'HC Angle'
                xlim = (-180, 180)
                xlabel = 'HC Angle (Degrees)'
                nbins = 180

            if key == 'Abs_HC_angle':
                xname = 'Abs HC angle >= ' + str(par['large_HC'])
                xfname = 'Abs HC angle < ' + str(par['large_HC'])
                xlabel = 'Abs HC Angle'
                x = np.rad2deg(np.abs(cur_cond.HC_angle))
                x_false = x[x < par['large_HC']]
                x_true = x[x >= par['large_HC']]
                xlabel = 'Abs HC Angle (Degrees)'
                xlim = (0, 180)
                nbins = 90

            if key == 'INS_interval':
                x = cur_cond.INS_interval
                x = x[~np.isnan(x)]
                xlabel = 'Inter-step-interval (s)'
                xlim = (0, 4)

            if key == 'INS_interval_no_HCs':
                x = cur_cond.INS_interval[cur_cond.next_event_is_step]
                xlabel = 'Inter-step-interval (s)'
                xlim = (0, 4)

            if key == 'back_vector_angular_speed':
                x = np.rad2deg(cur_cond.max_back_vector_angular_speed)
                x_false = np.rad2deg(
                    cur_cond.max_back_vector_angular_speed_false)
                xlabel = 'Back ang. speed (' + degree_sign + '/s)'
                xlim = (0, 100)

            # make hist
            edges = np.linspace(xlim[0], xlim[1], nbins + 1)
            hist = np.histogram(x, range=(np.nanmin(x), np.nanmax(x)),
                                bins=edges, density=False)[0]
            if df.empty:
                df = pandas.concat([df, pandas.DataFrame(edges),
                                    pandas.DataFrame(hist)],
                                   ignore_index=True, axis=1)
            else:
                df = pandas.concat([df, pandas.DataFrame(hist)],
                                   ignore_index=True, axis=1)

            title = key + " " + str(condition)
            # short_title = 'HC_' + dstrshort + '_box_d'
        print(df)
        if par['save_data']:
            title = "Histogram: " + key
            short_title = "Hist_" + key
            df.columns = ['x'] + conditions
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])

            self.excel_titles[short_title] = title

    def norm_individual_area_to_tracklength(
            self,
            par,
            condition,
            selected_trial_numbers,
            figure_name='tracks_on_dish'):

        # for all trials
        pref_trials = [
            self.dict[condition].trial[index] for index in sorted(
                np.unique(self.dict[condition].trial, return_index=True)[1])
        ]
        trial_numbers = [
            self.dict[condition].trial_number[index] for index in sorted(
                np.unique(
                    self.dict[condition].trial_number,
                    return_index=True)[1])
        ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        for trial_num in selected_trial_numbers:

            # plt.ion()
            norm_black_sum = 0.0
            # for all tracks
            trial = trial_dict[trial_num]
            # select trial
            trial_idx = self.dict[condition].trial == trial

            # for all tracks
            scond = self.dict[condition]
            trial_tracks = np.unique(scond.track_number[trial_idx])
            # start_time = timeit.default_timer()
            fig = plt.figure(
                figsize=(
                    par['fig_width'],
                    par['fig_width']))
            # fig.tight_layout(pad=0)
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
            for track_num in np.unique(scond.track_number[trial_idx]):
                plt.cla()
                plt.setp(ax1,
                         xlim=(-par['radius_dish'] - 5,
                               par['radius_dish'] + 5),
                         ylim=(-par['radius_dish'] - 5,
                               par['radius_dish'] + 5),
                         xticks=[], yticks=[])
                idx_track = scond.track_number[trial_idx] == track_num

                # Plot track
                ax1.plot(scond.spine4[trial_idx, 0][idx_track],
                         scond.spine4[trial_idx, 1][idx_track],
                         lw=par['area_line_width'], ls='-', color=(0, 0, 0))
                track_length = np.sum(
                    np.linalg.norm(
                        np.diff(scond.spine4[trial_idx][idx_track],
                                axis=0),
                        axis=1))
                fig.canvas.draw()
                im = np.fromstring(fig.canvas.tostring_rgb(),
                                   dtype=np.uint8, sep='')
                w = np.sqrt(im.shape[0] / 3)
                im = im.reshape((im.shape[0] / 3, 3))

                im = 1 * (np.sum(im, axis=1) < 1)
                black = im.sum()
                # print "Black: " + str(black)
                pixel_mm = 2 * par['radius_dish'] / w
                area = black * pixel_mm * pixel_mm
                norm_black_sum += area / (1.0 * track_length)

        norm_black_average = norm_black_sum / (1.0 * len(trial_tracks))
        plt.close()
        # print "NORM AVERAGE = " + str(norm_black_average)
        # print 'Trial: %.3f s' % (timeit.default_timer() -
        #                          start_time)
        return norm_black_average

    def group_area_trial_tracks_on_dish(
            self,
            par,
            condition,
            selected_trial_numbers,
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
        pref_trials = [
            self.dict[condition].trial[index] for index in sorted(
                np.unique(self.dict[condition].trial, return_index=True)[1])
        ]
        trial_numbers = [
            self.dict[condition].trial_number[index] for index in sorted(
                np.unique(
                    self.dict[condition].trial_number,
                    return_index=True)[1])
        ]
        trial_dict = dict(list(zip(trial_numbers, pref_trials)))
        for trial_num in selected_trial_numbers:
            trial = trial_dict[trial_num]
            # select trial
            trial_idx = self.dict[condition].trial == trial

            # for all tracks
            scond = self.dict[condition]
            for track_number in np.unique(
                    scond.track_number[trial_idx]):
                idx_track = scond.track_number[trial_idx] == track_number
                ax1.plot(scond.spine4[trial_idx, 0][idx_track],
                         scond.spine4[trial_idx, 1][idx_track],
                         lw=par['area_line_width'], ls='-', color=(0, 0, 0))

        fig.canvas.draw()
        im = np.fromstring(fig.canvas.tostring_rgb(),
                           dtype=np.uint8, sep='')
        im = im.reshape((im.shape[0] / 3, 3))
        black_im = 1 * (np.sum(im, axis=1) < 1)
        black = black_im.sum()
        plt.close()
        return (black * 1.0) / (1.0 * (im.shape[0]))

    def figure_group_area_explored_boxplot(self, par, conditions):

        # figure
        fig = plt.figure(
            figsize=(
                (0.55 + len(conditions) * 0.05) *
                par['fig_width'],
                (0.7 + len(conditions) * 0.05) *
                par['fig_width']))
        fig.subplots_adjust(left=0.35, right=0.96, hspace=0.3, wspace=0.4,
                            bottom=0.35, top=0.95)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        area_array = []

        for condition_idx, condition in enumerate(conditions):
            # init
            area_ratios = []
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # for all trials
            for trial_num in trial_numbers:
                areaval = self.group_area_trial_tracks_on_dish(par, condition,
                                                               [trial_num])
                area_ratios.append(areaval)
                area_array.append([
                    condition,
                    trial_dict[trial_num],
                    areaval])

            # convert to array
            area_ratios = np.array(area_ratios)

            # make boxplot
            bp = ax.boxplot(area_ratios, positions=[condition_idx],
                            widths=0.3, whis=1.6, sym='', patch_artist=True)
            plt.setp(bp['boxes'], color='k')
            plt.setp(bp['medians'], color='r')
            plt.setp(bp['caps'], color='k')
            plt.setp(bp['whiskers'], color='k', ls='-')
            plt.setp(bp['fliers'], color='k', marker='+')

            for patch in bp['boxes']:
                patch.set_facecolor('white')

        # figure settings (has to come after boxplot)
        plt.setp(ax, ylabel='Group Area Covered',
                 xlim=(-0.5, len(conditions) - 0.5),
                 ylim=(0.0, 1.0), xticks=list(range(len(conditions))),
                 yticks=[x / 10.0 for x in range(0, 11, 1)])
        ax.set_xticklabels(
            [self.names_short[condition_idx]
             for condition_idx, condition in enumerate(conditions)],
            rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_ticks_position('left')
        # ax.set_ticks = [x/10.0 for x in range(-10, 11, 2)]

        # ax.axhline(0, color='gray', zorder=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(area_array)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "Area Covered"]
            df.to_excel(self.excelWriter,
                        sheet_name='Group Area Covered',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'Group Area Covered'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' +
                        'boxplot_Group_Area-' + str(par['area_line_width'])
                        )
            plt.close()

    def figure_norm_track_area_explored_boxplot(self, par, conditions):

        # figure
        fig = plt.figure(
            figsize=(
                (0.55 + len(conditions) * 0.05) *
                par['fig_width'],
                (0.7 + len(conditions) * 0.05) *
                par['fig_width']))
        fig.subplots_adjust(left=0.35, right=0.96, hspace=0.3, wspace=0.4,
                            bottom=0.35, top=0.95)

        # figure settings
        ax = fig.add_subplot(111)
        [ax.spines[str_tmp].set_color('none') for str_tmp in ['top', 'right']]
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        area_array = []

        for condition_idx, condition in enumerate(conditions):
            # init
            area_ratios = []
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            try:
                area_norm_rat = self.dict[condition].area_averages
                for areaval in area_norm_rat:
                    area_ratios.append(areaval)
                    area_array.append([
                        condition,
                        trial_dict[1],  # TODO: Something is weird here!!
                        areaval])
            except:
                # for all trials
                for trial_num in trial_numbers:
                    areaval = self.norm_individual_area_to_tracklength(
                        par,
                        condition,
                        [trial_num])
                    area_ratios.append(areaval)
                    area_array.append([
                        condition,
                        trial_dict[trial_num],
                        areaval])

            # convert to array
            area_ratios = np.array(area_ratios)

            # make boxplot
            bp = ax.boxplot(area_ratios, positions=[condition_idx],
                            widths=0.3, whis=1.6, sym='', patch_artist=True)
            plt.setp(bp['boxes'], color='k')
            plt.setp(bp['medians'], color='r')
            plt.setp(bp['caps'], color='k')
            plt.setp(bp['whiskers'], color='k', ls='-')
            plt.setp(bp['fliers'], color='k', marker='+')

            for patch in bp['boxes']:
                patch.set_facecolor('white')

        # figure settings (has to come after boxplot)
        plt.setp(ax, ylabel='Normalized Area Covered',
                 xlim=(-0.5, len(conditions) - 0.5),
                 # ylim=(0.0, 1.0),
                 xticks=list(range(len(conditions))),
                 # yticks=[x/10.0 for x in range(0, 11, 1)]
                 )
        ax.set_xticklabels(
            [self.names_short[condition_idx]
             for condition_idx, condition in enumerate(conditions)],
            rotation=45, ha='right', size=font_size)
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_ticks_position('left')
        # ax.set_ticks = [x/10.0 for x in range(-10, 11, 2)]

        # ax.axhline(0, color='gray', zorder=0)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(area_array)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "Norm Area Covered"]
            df.to_excel(self.excelWriter,
                        sheet_name='Normalized Area Covered',
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'Normalized Area Covered'])

        # save plot
        if par['save_figure']:
            plt.savefig(par['figure_dir'] +
                        '/' +
                        'boxplot_norm_area-' + str(par['area_line_width'])
                        )
            plt.close()

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

        for condition_idx, condition in enumerate(conditions):
            condition_data = self.dict[condition]
            # init
            pref_trials = [
                condition_data.trial[index] for index in sorted(
                    np.unique(condition_data.trial, return_index=True)[1])
            ]
            trial_numbers = [
                condition_data.trial_number[index] for index in sorted(
                    np.unique(
                        condition_data.trial_number, return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            full_condition = condition_data.full_condition

            HC_start_idx = condition_data.HC_start_idx
            HC_end_idx = condition_data.HC_end_idx
            HC_angle = condition_data.HC_angle

            if (bearing_limited == 'l' or bearing_limited == 'il'):
                bearing_angle_idx = np.array([x <= np.deg2rad(135) and x >= np.deg2rad(45)
                                              or x >= np.deg2rad(-135) and x <= np.deg2rad(-45) for x in
                                              condition_data.bearing_angle])[
                    condition_data.HC_start_idx]
                if (bearing_limited == 'il'):
                    bearing_angle_idx = ~bearing_angle_idx
                HC_start_idx = HC_start_idx[bearing_angle_idx]
                HC_end_idx = HC_end_idx[bearing_angle_idx]
                HC_angle = HC_angle[bearing_angle_idx]

            large_HC_bidx = angleComp(HC_angle, large_HC,
                                      subthreshold)

            for trial_number in trial_numbers:

                idx_trial = (
                        condition_data.trial_number
                        [HC_start_idx[large_HC_bidx]] ==
                        trial_number)
                idx_trial_at_end = (
                        condition_data.trial_number
                        [HC_end_idx[large_HC_bidx]] ==
                        trial_number)

                heading_angle_at_start = condition_data.heading_angle[
                    HC_start_idx[large_HC_bidx]][idx_trial]
                heading_angle_at_end = condition_data.heading_angle[
                    HC_end_idx[large_HC_bidx]][
                    idx_trial_at_end]

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
                    full_condition,
                    trial_dict[trial_number],
                    full_condition,
                    weight_average,
                ])

        # save data
        if par['save_data']:

            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          'Conditions',
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
            blist = createPlotlyBoxPlot(
                df=df,
                title="Boxplot: HC " + dstr + long_lsuffix,
                description=description,
                values='value',
                label="Group",
                do_stats=True,
                base=base,
                include_plotly=(not self.plotly_included)
            )
            if not self.plotly_included:
                self.plotly_included = True
            self.plotly_columns = self.plotly_columns + blist
            df.to_excel(self.excelWriter,
                        sheet_name=short_title,
                        index=False)
            fixXLColumns(
                df,
                self.excelWriter.sheets[short_title])
            self.excel_titles[short_title] = title

    def figure_HC_reorientation_boxplot_distance_split(
            self,
            par,
            conditions,
            distance,
            reorientation=True,
            large_HC=-1,
            subthreshold=False,
            bearing_limited='nl',
            subtract_mean=False,
            mean_of_bins=False,
            heading=''
    ):

        sub_string = ""
        sub_str = ""
        lsuffix = ''
        long_lsuffix = ''
        mb_suffix = ''
        long_mbsuffix = ''

        edges_bearing = np.linspace(-np.pi, np.pi, 20)

        if (subtract_mean):
            sub_string = "subtract mean"
            sub_str = "_sm"

        if (bearing_limited == 'l'):
            lsuffix = "_l"
            long_lsuffix = 'limited'
        if (bearing_limited == 'il'):
            lsuffix = "_il"
            long_lsuffix = 'inverse limited'

        if (mean_of_bins):
            mbsuffix = "_mb"
            long_mbsuffix = 'mean of bins'

        if (large_HC == -1):
            large_HC = par['large_HC']
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
            angle_str = " (min angle:" + str(large_HC) + ") "
            dstrshort = dstr + "_minA" + str(int(large_HC))
            dstr = dstr + angle_str

        # column_names = ['Reorientation/near', 'reorientation/near',
        #                 'Reorientation/far', 'reorientation/far']
        # if(distance == 0):
        #     column_names = [dstr]
        # else:
        #     column_names = [dstr + '/near',
        #                     dstr + '/far']
        # column_names_short = ['towards', 'away']

        all_weights = self.reorientation_minus_mean(
            par=par,
            conditions=conditions,
            distance=distance,
            large_HC=large_HC,
            reorientation=True,
            subthreshold=subthreshold
        )

        if (distance > 0):
            far_all_weights = all_weights[0]
            near_all_weights = all_weights[1]
        else:
            far_all_weights = all_weights

        boxplot_data = []

        for condition_idx, condition in enumerate(conditions):
            condition_data = self.dict[condition]
            # init
            pref_trials = [
                condition_data.trial[index] for index in sorted(
                    np.unique(condition_data.trial, return_index=True)[1])
            ]
            trial_numbers = [
                condition_data.trial_number[index] for index in sorted(
                    np.unique(
                        condition_data.trial_number, return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            full_condition = condition_data.full_condition

            # butterfly filtering of bearing angle
            bearing_angle_idx_l = np.array([x < np.deg2rad(135) and x > np.deg2rad(45)
                                            or x > np.deg2rad(-135) and x < np.deg2rad(-45) for x in
                                            condition_data.bearing_angle])[
                condition_data.HC_start_idx]

            # hourglass filtering of bearing angle
            bearing_angle_idx_il = np.array([x < np.deg2rad(45) and x > np.deg2rad(-45)
                                             or x > np.deg2rad(135) and x < np.deg2rad(-135) for x in
                                             condition_data.bearing_angle])[
                condition_data.HC_start_idx]

            # heading towards
            bearing_angle_idx_towards = np.array([abs(x) < np.deg2rad(90)
                                                  for x in condition_data.bearing_angle])[
                condition_data.HC_start_idx]
            # heading away
            bearing_angle_idx_away = np.array([abs(x) > np.deg2rad(90)
                                               for x in condition_data.bearing_angle])[
                condition_data.HC_start_idx]

            if distance > 0:
                bearing_idx_near_l = bearing_angle_idx_l[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                bearing_idx_near_il = bearing_angle_idx_il[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]

                bearing_idx_near_towards = bearing_angle_idx_towards[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                bearing_idx_near_away = bearing_angle_idx_away[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]

                near_HC_start_idx = condition_data.HC_start_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_end_idx = condition_data.HC_end_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_angle = condition_data.HC_angle[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                # near_large_HC_bidx = np.abs(
                #     near_HC_angle) > np.deg2rad(angle)

                if (bearing_limited == 'l'):
                    near_HC_start_idx = near_HC_start_idx[bearing_idx_near_l]
                    near_HC_end_idx = near_HC_end_idx[bearing_idx_near_l]
                    near_HC_angle = near_HC_angle[bearing_idx_near_l]
                if (bearing_limited == 'il'):
                    near_HC_start_idx = near_HC_start_idx[bearing_idx_near_il]
                    near_HC_end_idx = near_HC_end_idx[bearing_idx_near_il]
                    near_HC_angle = near_HC_angle[bearing_idx_near_il]

                if (heading == 'towards'):
                    near_HC_start_idx = near_HC_start_idx[bearing_idx_near_towards]
                    near_HC_end_idx = near_HC_end_idx[bearing_idx_near_towards]
                    near_HC_angle = near_HC_angle[bearing_idx_near_towards]
                if (heading == 'away'):
                    near_HC_start_idx = near_HC_start_idx[bearing_idx_near_away]
                    near_HC_end_idx = near_HC_end_idx[bearing_idx_near_away]
                    near_HC_angle = near_HC_angle[bearing_idx_near_away]

                near_large_HC_bidx = angleComp(near_HC_angle, large_HC,
                                               subthreshold)

            bearing_idx_far_l = bearing_angle_idx_l[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            bearing_idx_far_il = bearing_angle_idx_il[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            bearing_idx_far_towards = bearing_angle_idx_towards[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            bearing_idx_far_away = bearing_angle_idx_away[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]

            far_HC_start_idx = condition_data.HC_start_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]

            far_HC_end_idx = condition_data.HC_end_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_angle = condition_data.HC_angle[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            # far_large_HC_bidx = np.abs(
            #     far_HC_angle) > np.deg2rad(angle)

            if (bearing_limited == 'l'):
                far_HC_start_idx = far_HC_start_idx[bearing_idx_far_l]
                far_HC_end_idx = far_HC_end_idx[bearing_idx_far_l]
                far_HC_angle = far_HC_angle[bearing_idx_far_l]

            if (bearing_limited == 'il'):
                far_HC_start_idx = far_HC_start_idx[bearing_idx_far_il]
                far_HC_end_idx = far_HC_end_idx[bearing_idx_far_il]
                far_HC_angle = far_HC_angle[bearing_idx_far_il]
            if (heading == 'towards'):
                far_HC_start_idx = far_HC_start_idx[bearing_idx_far_towards]
                far_HC_end_idx = far_HC_end_idx[bearing_idx_far_towards]
                far_HC_angle = far_HC_angle[bearing_idx_far_towards]

            if (heading == 'away'):
                far_HC_start_idx = far_HC_start_idx[bearing_idx_far_away]
                far_HC_end_idx = far_HC_end_idx[bearing_idx_far_away]
                far_HC_angle = far_HC_angle[bearing_idx_far_away]

            far_large_HC_bidx = angleComp(far_HC_angle, large_HC,
                                          subthreshold)

            far_weights_new = far_all_weights[condition_idx]
            if (distance > 0):
                near_weights_new = near_all_weights[condition_idx]

            # for all trials
            # print "========= NEAR WEIGHTS " +
            # dstr + " d: = " + str(distance) +
            # "condition: " + str(condition) + "============"

            for trial_number in trial_numbers:

                if distance > 0:
                    near_idx_trial = (
                            condition_data.trial_number
                            [near_HC_start_idx[near_large_HC_bidx]] ==
                            trial_number)
                    near_idx_trial_at_end = (
                            condition_data.trial_number
                            [near_HC_end_idx[near_large_HC_bidx]] ==
                            trial_number)

                    near_bearing_angle_at_start = condition_data.bearing_angle[
                        near_HC_start_idx[near_large_HC_bidx]][
                        near_idx_trial]

                    near_heading_angle_at_start = condition_data.heading_angle[
                        near_HC_start_idx[near_large_HC_bidx]][near_idx_trial]
                    near_heading_angle_at_end = condition_data.heading_angle[
                        near_HC_end_idx[near_large_HC_bidx]][
                        near_idx_trial_at_end]

                far_idx_trial = (
                        condition_data.trial_number
                        [far_HC_start_idx[far_large_HC_bidx]] ==
                        trial_number)

                far_idx_trial_at_end = (
                        condition_data.trial_number
                        [far_HC_end_idx[far_large_HC_bidx]] ==
                        trial_number)
                far_bearing_angle_at_start = condition_data.bearing_angle[
                    far_HC_start_idx[far_large_HC_bidx]][
                    far_idx_trial]
                far_heading_angle_at_start = condition_data.heading_angle[
                    far_HC_start_idx[far_large_HC_bidx]][far_idx_trial]
                far_heading_angle_at_end = condition_data.heading_angle[
                    far_HC_end_idx[far_large_HC_bidx]][
                    far_idx_trial_at_end]

                if reorientation:

                    if distance > 0:
                        near_weights = np.rad2deg(
                            np.abs(near_heading_angle_at_start) -
                            np.abs(near_heading_angle_at_end))
                        # np.abs(near_bearing_angle_at_start) -
                        # np.abs(near_heading_angle_at_end))
                        if (len(near_weights) != 0):
                            near_hist = binned_statistic(near_bearing_angle_at_start, near_weights, bins=edges_bearing)[
                                0]
                            near_mean = np.nanmean(near_hist[~np.isnan(near_hist)])
                    far_weights = np.rad2deg(
                        np.abs(far_heading_angle_at_start) -
                        np.abs(far_heading_angle_at_end))
                    # np.abs(far_bearing_angle_at_start) -
                    # np.abs(far_heading_angle_at_end))

                    if (len(far_weights) != 0):
                        far_hist = binned_statistic(far_bearing_angle_at_start, far_weights, bins=edges_bearing)[0]

                        far_mean = np.mean(far_hist[~np.isnan(far_hist)])

                else:
                    if distance > 0:
                        near_weights = np.rad2deg(
                            np.abs(near_heading_angle_at_end))
                    far_weights = np.rad2deg(
                        np.abs(far_heading_angle_at_end))

                if (subtract_mean):
                    if distance > 0:
                        if (len(near_weights_new) != len(near_idx_trial)):
                            near_weights_new = near_weights_new[:len(near_idx_trial)]
                    if (len(far_weights_new) != len(far_idx_trial)):
                        far_weights_new = far_weights_new[:len(far_idx_trial)]
                    if reorientation:
                        if distance > 0:
                            near_weights = np.array(near_weights_new)[near_idx_trial]

                        far_weights = np.array(far_weights_new)[far_idx_trial]

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

                # print np.sum(near_weights)
                # print len(near_weights)
                # print "Trial: " + str(trial_number)
                # print near_weights.shape
                if (not (mean_of_bins)):

                    if distance > 0:
                        weight_average_near = (
                                np.sum(near_weights) / len(near_weights))

                    weight_average_far = (
                            np.sum(far_weights) / len(far_weights)
                    )

                    if distance > 0:
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
                else:
                    if distance > 0:
                        boxplot_data.append([
                            full_condition,
                            trial_dict[trial_number],
                            full_condition + ' near',
                            near_mean,
                        ])

                    boxplot_data.append([
                        full_condition,
                        trial_dict[trial_number],
                        full_condition + ' far',
                        far_mean,
                    ])

        # save data
        if par['save_data']:

            df = pandas.DataFrame(boxplot_data)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group",
                          "Trial Name",
                          'Conditions',
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

            if distance > 0:
                title = "Boxplot: HC " + dstr + ' split distance ' + long_lsuffix + sub_string + long_mbsuffix + '_' + heading
                short_title = 'HC_' + dstrshort + '_box_d' + lsuffix + sub_str + mb_suffix + '_' + heading
                blist_far = createPlotlyBoxPlot(
                    df=df[df['Conditions'].str.contains('far')],
                    title="Boxplot: HC " + dstr + " far " + long_lsuffix + " " + sub_string + long_mbsuffix + '_' + heading,
                    description=description + " Distance > " +
                                str(distance),
                    values='value',
                    label="Conditions",
                    base=base,
                    do_stats=True,
                    include_plotly=(not self.plotly_included)
                )
                if not self.plotly_included:
                    self.plotly_included = True

                blist_near = createPlotlyBoxPlot(
                    df=df[df['Conditions'].str.contains('near')],
                    title="Boxplot: HC " + dstr + " near " + long_lsuffix + " " + sub_string + long_mbsuffix + '_' + heading,
                    description=description + " Distance < " +
                                str(distance),
                    values='value',
                    base=base,
                    label="Conditions",
                    do_stats=True)
                self.plotly_columns = self.plotly_columns + blist_far
                self.plotly_columns = self.plotly_columns + blist_near
                df.to_excel(self.excelWriter,
                            sheet_name=short_title,
                            index=False)
                fixXLColumns(
                    df,
                    self.excelWriter.sheets[short_title])
                self.excel_titles[short_title] = title
            else:
                title = "Boxplot: HC " + dstr + long_lsuffix + " " + sub_string + long_mbsuffix + '_' + heading
                short_title = 'HC_' + dstrshort + '_box' + lsuffix + sub_str + mb_suffix + '_' + heading
                blist = createPlotlyBoxPlot(
                    df=df,
                    title="Boxplot: HC " + dstr + long_lsuffix + " " + sub_string + long_mbsuffix + '_' + heading,
                    description=description,
                    values='value',
                    label="Group",
                    do_stats=True,
                    base=base,
                    include_plotly=(not self.plotly_included)
                )
                if not self.plotly_included:
                    self.plotly_included = True
                self.plotly_columns = self.plotly_columns + blist
                df.to_excel(self.excelWriter,
                            sheet_name=short_title,
                            index=False)
                fixXLColumns(
                    df,
                    self.excelWriter.sheets[short_title])
                self.excel_titles[short_title] = title

    def HC_reorientation_to_bearing(
            self,
            par,
            conditions,
            distance,
            reorientation=True,
            large_HC=-1,
            subthreshold=False,

    ):

        line_data = np.array([])
        line_data_new = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        line_data_new = np.rad2deg(edges_bearing)[:-1]

        far_hist_sum = np.zeros(99)
        near_hist_sum = np.zeros(99)

        far_all_weights = []
        far_all_bearing_at_start = []

        near_all_weights = []
        near_all_bearing_at_start = []

        for condition_idx, condition in enumerate(conditions):
            condition_data = self.dict[condition]

            if distance > 0:
                near_HC_start_idx = condition_data.HC_start_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_end_idx = condition_data.HC_end_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_angle = condition_data.HC_angle[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_large_HC_bidx = angleComp(near_HC_angle, large_HC,
                                               subthreshold)

            far_HC_start_idx = condition_data.HC_start_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_end_idx = condition_data.HC_end_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_angle = condition_data.HC_angle[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_large_HC_bidx = angleComp(far_HC_angle, large_HC,
                                          subthreshold)

            if distance > 0:
                near_bearing_angle_at_start = condition_data.bearing_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_start = condition_data.heading_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_end = condition_data.heading_angle[
                    near_HC_end_idx[near_large_HC_bidx]]

            far_bearing_angle_at_start = condition_data.bearing_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_start = condition_data.heading_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_end = condition_data.heading_angle[
                far_HC_end_idx[far_large_HC_bidx]]

            if reorientation:
                if distance > 0:
                    near_weights = np.rad2deg(
                        np.abs(near_heading_angle_at_start) -
                        np.abs(near_heading_angle_at_end))
                    # np.abs(near_bearing_angle_at_start) -
                    # np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_start) -
                    np.abs(far_heading_angle_at_end))
                # np.abs(far_bearing_angle_at_start) -
                # np.abs(far_heading_angle_at_end))
            else:
                if distance > 0:
                    near_weights = np.rad2deg(
                        np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_end))

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

            # print np.sum(near_weights)
            # print len(near_weights)
            # print "Trial: " + str(trial_number)
            # print near_weights.shape
            # add data for circular boundary conditions
            if distance > 0:
                near_bearing_angle_at_start = np.hstack(
                    [
                        near_bearing_angle_at_start - 2 * np.pi,
                        near_bearing_angle_at_start,
                        near_bearing_angle_at_start + 2 * np.pi])
                near_weights = np.tile(near_weights, 3)
                near_all_weights.append(near_weights)
                near_all_bearing_at_start.append(near_bearing_angle_at_start)

            far_bearing_angle_at_start = np.hstack(
                [far_bearing_angle_at_start - 2 * np.pi,
                 far_bearing_angle_at_start,
                 far_bearing_angle_at_start + 2 * np.pi])
            far_weights = np.tile(far_weights, 3)
            far_all_weights.append(far_weights)
            far_all_bearing_at_start.append(far_bearing_angle_at_start)

        far_all_weights = np.concatenate(far_all_weights)
        far_all_bearing_at_start = np.concatenate(far_all_bearing_at_start)
        if distance > 0:
            near_all_weights = np.concatenate(near_all_weights)
            near_all_bearing_at_start = np.concatenate(near_all_bearing_at_start)

        return far_all_weights, far_all_bearing_at_start, near_all_weights, near_all_bearing_at_start

    def reorientation_minus_mean(
            self,
            par,
            conditions,
            distance,
            reorientation=True,
            large_HC=-1,
            subthreshold=False

    ):
        if (large_HC == -1):
            large_HC = par['large_HC']

        if reorientation:
            angle = par['large_HC']

        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        line_data_ = np.array([])

        line_data_ = np.rad2deg(edges_bearing)[:-1]

        edges_bearing = 99

        all_pooled = self.HC_reorientation_to_bearing(
            par,
            conditions=conditions,
            distance=distance,
            reorientation=reorientation,
            large_HC=large_HC,
            subthreshold=subthreshold
        )
        far_all_weights = all_pooled[0]
        far_all_bearing_at_start = all_pooled[1]
        near_all_weights = all_pooled[2]
        near_all_bearing_at_start = all_pooled[3]

        far_hist_mean = binned_statistic(far_all_bearing_at_start, far_all_weights, bins=edges_bearing)[0]
        far_hist_ind = binned_statistic(far_all_bearing_at_start, far_all_weights, bins=edges_bearing)[2]
        if distance > 0:
            near_hist_mean = binned_statistic(near_all_bearing_at_start, near_all_weights, bins=edges_bearing)[0]
            near_hist_ind = binned_statistic(near_all_bearing_at_start, near_all_weights, bins=edges_bearing)[2]
        # near_hist_mean=np.convolve(np.ones(11) / 11.,near_hist_mean, mode='same')

        # far_hist_mean=np.convolve(np.ones(11) / 11.,far_hist_mean, mode='same')

        far_all_weights_new = []
        near_all_weights_new = []
        for condition_idx, condition in enumerate(conditions):
            condition_data = self.dict[condition]

            if distance > 0:
                near_HC_start_idx = condition_data.HC_start_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_end_idx = condition_data.HC_end_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_angle = condition_data.HC_angle[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_large_HC_bidx = angleComp(near_HC_angle, large_HC,
                                               subthreshold)

            far_HC_start_idx = condition_data.HC_start_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_end_idx = condition_data.HC_end_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_angle = condition_data.HC_angle[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_large_HC_bidx = angleComp(far_HC_angle, large_HC,
                                          subthreshold)

            if distance > 0:
                near_bearing_angle_at_start = condition_data.bearing_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_start = condition_data.heading_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_end = condition_data.heading_angle[
                    near_HC_end_idx[near_large_HC_bidx]]

            far_bearing_angle_at_start = condition_data.bearing_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_start = condition_data.heading_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_end = condition_data.heading_angle[
                far_HC_end_idx[far_large_HC_bidx]]

            if distance > 0:
                near_weights = np.rad2deg(
                    np.abs(near_heading_angle_at_start) -
                    np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_start) -
                    np.abs(far_heading_angle_at_end))
            far_weights = np.rad2deg(
                np.abs(far_heading_angle_at_start) -
                np.abs(far_heading_angle_at_end))
            if distance > 0:
                near_bearing_angle_at_start = np.hstack(
                    [
                        near_bearing_angle_at_start - 2 * np.pi,
                        near_bearing_angle_at_start,
                        near_bearing_angle_at_start + 2 * np.pi])
                near_weights = np.tile(near_weights, 3)

            far_bearing_angle_at_start = np.hstack(
                [far_bearing_angle_at_start - 2 * np.pi,
                 far_bearing_angle_at_start,
                 far_bearing_angle_at_start + 2 * np.pi])
            far_weights = np.tile(far_weights, 3)
            if distance > 0:

                near_weights_new = []
                near_hist_ind = binned_statistic(near_bearing_angle_at_start, near_weights, bins=edges_bearing)[2]
                for n, weight in enumerate(near_weights):
                    near_weights_new.append(weight - near_hist_mean[near_hist_ind[n] - 1])

                near_n_samples_new = np.histogram(near_bearing_angle_at_start,
                                                  bins=edges_bearing,
                                                  normed=False)[0]
                near_hist_new = np.histogram(near_bearing_angle_at_start,
                                             bins=edges_bearing,
                                             normed=False,
                                             weights=near_weights_new)[0]
                near_hist_new = near_hist_new / near_n_samples_new
                near_hist_new = np.convolve(np.ones(11) / 11., near_hist_new, mode='same')

                line_data_ = np.vstack((line_data_, near_hist_new))
                near_all_weights_new.append(near_weights_new)
            far_weights_new = []

            far_hist_ind = binned_statistic(far_bearing_angle_at_start, far_weights, bins=edges_bearing)[2]
            for n, weight in enumerate(far_weights):
                far_weights_new.append(weight - far_hist_mean[far_hist_ind[n] - 1])
            far_all_weights_new.append(far_weights_new)  # [:len(far_weights_new)/3])

        if (distance > 0):
            return far_all_weights_new, near_all_weights_new
        else:
            return far_all_weights_new

    def figure_HC_reorientation_to_bearing_distance_split(
            self,
            par,
            conditions,
            distance,
            reorientation=True,
            large_HC=-1,
            subthreshold=False,
            subtract_mean=False,
            mean_of_bins=False
    ):

        if (large_HC == -1):
            large_HC = par['large_HC']

        if reorientation:
            angle = par['large_HC']
            dstr = 'reor'
            ylabel = "HC reorientation"
            yrange = (-30, 30)
            yt = np.arange(-30, 31, 5)
            hline_at = 0
        else:
            angle = par['large_HC']
            dstr = 'accuracy'
            ylabel = "HC accuracy"
            yrange = (60, 120)
            yt = np.arange(60, 121, 10)
            hline_at = 90

        if subthreshold:
            angle_str = " (max angle:" + str(large_HC) + ")"
            dstrshort = dstr + "_maxA" + str(int(large_HC))
            dstr = dstr + angle_str
        else:
            angle_str = " (min angle:" + str(large_HC) + ")"
            dstrshort = dstr + "_minA" + str(int(large_HC))
            dstr = dstr + angle_str

        # column_names = ['Reorientation/near', 'reorientation/near',
        #                 'Reorientation/far', 'reorientation/far']
        # if(distance == 0):
        #     column_names = [dstr]
        # else:
        #     column_names = [dstr + '(near)',
        #                     dstr + '(far)']
        # column_names_short = ['towards', 'away']

        line_data = np.array([])
        edges_bearing = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 100)
        line_data = np.rad2deg(edges_bearing)[:-1]
        line_data_ = np.array([])

        line_data_ = np.rad2deg(edges_bearing)[:-1]

        # edges_bearing = 99

        all_pooled = self.HC_reorientation_to_bearing(
            par,
            conditions=conditions,
            distance=distance,
            reorientation=reorientation,
            large_HC=large_HC,
            subthreshold=subthreshold
        )
        far_all_weights = all_pooled[0]
        far_all_bearing_at_start = all_pooled[1]
        near_all_weights = all_pooled[2]
        near_all_bearing_at_start = all_pooled[3]

        far_hist_mean = binned_statistic(far_all_bearing_at_start, far_all_weights, bins=edges_bearing)[0]
        far_hist_ind = binned_statistic(far_all_bearing_at_start, far_all_weights, bins=edges_bearing)[2]
        if distance > 0:
            near_hist_mean = binned_statistic(near_all_bearing_at_start, near_all_weights, bins=edges_bearing)[0]
            near_hist_ind = binned_statistic(near_all_bearing_at_start, near_all_weights, bins=edges_bearing)[2]
        # near_hist_mean=np.convolve(np.ones(11) / 11.,near_hist_mean, mode='same')

        # far_hist_mean=np.convolve(np.ones(11) / 11.,far_hist_mean, mode='same')

        sub_string = ""
        sub_str = ""

        for condition_idx, condition in enumerate(conditions):
            condition_data = self.dict[condition]

            if distance > 0:
                near_HC_start_idx = condition_data.HC_start_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_end_idx = condition_data.HC_end_idx[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_HC_angle = condition_data.HC_angle[
                    condition_data.distance[
                        condition_data.HC_start_idx] < distance]
                near_large_HC_bidx = angleComp(near_HC_angle, large_HC,
                                               subthreshold)

            far_HC_start_idx = condition_data.HC_start_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_end_idx = condition_data.HC_end_idx[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_HC_angle = condition_data.HC_angle[
                condition_data.distance[
                    condition_data.HC_start_idx] >= distance]
            far_large_HC_bidx = angleComp(far_HC_angle, large_HC,
                                          subthreshold)

            if distance > 0:
                near_bearing_angle_at_start = condition_data.bearing_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_start = condition_data.heading_angle[
                    near_HC_start_idx[near_large_HC_bidx]]
                near_heading_angle_at_end = condition_data.heading_angle[
                    near_HC_end_idx[near_large_HC_bidx]]

            far_bearing_angle_at_start = condition_data.bearing_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_start = condition_data.heading_angle[
                far_HC_start_idx[far_large_HC_bidx]]
            far_heading_angle_at_end = condition_data.heading_angle[
                far_HC_end_idx[far_large_HC_bidx]]

            if reorientation:
                if distance > 0:
                    near_weights = np.rad2deg(
                        np.abs(near_heading_angle_at_start) -
                        np.abs(near_heading_angle_at_end))
                    # np.abs(near_bearing_angle_at_start) -
                    # np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_start) -
                    np.abs(far_heading_angle_at_end))
                # np.abs(far_bearing_angle_at_start) -
                # np.abs(far_heading_angle_at_end))
            else:
                if distance > 0:
                    near_weights = np.rad2deg(
                        np.abs(near_heading_angle_at_end))

                far_weights = np.rad2deg(
                    np.abs(far_heading_angle_at_end))

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

            # print np.sum(near_weights)
            # print len(near_weights)
            # print "Trial: " + str(trial_number)
            # print near_weights.shape
            # add data for circular boundary conditions
            if distance > 0:
                near_bearing_angle_at_start = np.hstack(
                    [
                        near_bearing_angle_at_start - 2 * np.pi,
                        near_bearing_angle_at_start,
                        near_bearing_angle_at_start + 2 * np.pi])
                near_weights = np.tile(near_weights, 3)

            far_bearing_angle_at_start = np.hstack(
                [far_bearing_angle_at_start - 2 * np.pi,
                 far_bearing_angle_at_start,
                 far_bearing_angle_at_start + 2 * np.pi])
            far_weights = np.tile(far_weights, 3)

            # hist
            if distance > 0:
                near_n_samples = np.histogram(near_bearing_angle_at_start,
                                              bins=edges_bearing,
                                              normed=False)[0]
                near_hist = np.histogram(near_bearing_angle_at_start,
                                         bins=edges_bearing,
                                         normed=False,
                                         weights=near_weights)[0]
                near_hist = np.divide(near_hist, near_n_samples)

                # convolve, filter width = 60 degree
                near_hist = np.convolve(np.ones(11) / 11.,
                                        near_hist, mode='same')
                line_data = np.vstack((line_data, near_hist))

            far_n_samples = np.histogram(far_bearing_angle_at_start,
                                         bins=edges_bearing,
                                         normed=False)[0]
            far_hist = np.histogram(far_bearing_angle_at_start,
                                    bins=edges_bearing,
                                    normed=False,
                                    weights=far_weights)[0]

            far_hist = far_hist / far_n_samples

            # convolve, filter width = 60 degree
            far_hist = np.convolve(np.ones(11) / 11.,
                                   far_hist, mode='same')

            line_data = np.vstack((line_data, far_hist))

            if (subtract_mean):
                sub_string = "subtract mean"
                sub_str = "s_m"
                if distance > 0:

                    near_weights_new = []
                    near_hist_ind = binned_statistic(near_bearing_angle_at_start, near_weights, bins=edges_bearing)[2]
                    for n, weight in enumerate(near_weights):
                        near_weights_new.append(weight - near_hist_mean[near_hist_ind[n] - 1])

                    near_n_samples_new = np.histogram(near_bearing_angle_at_start,
                                                      bins=edges_bearing,
                                                      normed=False)[0]
                    near_hist_new = np.histogram(near_bearing_angle_at_start,
                                                 bins=edges_bearing,
                                                 normed=False,
                                                 weights=near_weights_new)[0]
                    near_hist_new = near_hist_new / near_n_samples_new
                    near_hist_new = np.convolve(np.ones(11) / 11., near_hist_new, mode='same')

                    line_data_ = np.vstack((line_data_, near_hist_new))

                far_weights_new = []

                far_hist_ind = binned_statistic(far_bearing_angle_at_start, far_weights, bins=edges_bearing)[2]
                for n, weight in enumerate(far_weights):
                    far_weights_new.append(weight - far_hist_mean[far_hist_ind[n] - 1])
                far_n_samples_new = np.histogram(far_bearing_angle_at_start,
                                                 bins=edges_bearing,
                                                 normed=False)[0]
                far_hist_new = np.histogram(far_bearing_angle_at_start,
                                            bins=edges_bearing,
                                            normed=False,
                                            weights=far_weights_new)[0]
                far_hist_new = far_hist_new / far_n_samples_new

                # convolve, filter width = 60 degree
                far_hist_new = np.convolve(np.ones(11) / 11., far_hist_new, mode='same')

                line_data_ = np.vstack((line_data_, far_hist_new))

        # save data
        if par['save_data']:
            if distance > 0:
                cur_column_names = [
                    x + '/' + str(y) for x in list(conditions) for y in [
                        'near', 'far']]
            else:
                cur_column_names = list(conditions)

            cur_column_names.insert(0, 'Bearing Angle')
            saved_line_idx = np.abs(line_data[0]) <= 183
            saved_line_idx_ = np.abs(line_data_[0]) <= 183

            df = pandas.DataFrame(line_data.T[saved_line_idx])
            if (subtract_mean):
                df = pandas.DataFrame(line_data_.T[saved_line_idx_])

            # print line_data.shape
            # print cur_column_names
            # if(subtract_mean):
            # cur_column_names = ['A','B' ]
            df.columns = cur_column_names

            if reorientation:
                description = """
                HC reorientation is the
                abs(Bearing angle)-abs(Heading angle at end of HC). Which
                shows how much the bearing is improved by each HC.
                """

            else:
                description = """
                HC Accuracy is the abs(HC angle at end of HC). How well
                directed is each HC.
                """

            if distance > 0:
                title = ("LinePlot: HC " + dstr +
                         " to bearing split distance " + sub_string)
                short_title = 'HC_' + dstrshort + '_line_d' + sub_str
                blist_near = createPlotlyLinePlot(
                    df=df.ix[:, [0] + list(range(1, len(cur_column_names), 2))],
                    xrange=[-180, 180],
                    title="HC " + dstr +
                          " to bearing near " + sub_string,
                    description=description,
                    include_plotly=(not self.plotly_included)
                )
                if not self.plotly_included:
                    self.plotly_included = True
                self.plotly_columns = self.plotly_columns + blist_near
                blist_far = createPlotlyLinePlot(
                    df=df.ix[:, [0] + list(range(2, len(cur_column_names), 2))],
                    xrange=[-180, 180],
                    title="HC " + dstr +
                          " to bearing far " + sub_string,
                    description=description)
                self.plotly_columns = self.plotly_columns + blist_far
                df.to_excel(self.excelWriter,
                            sheet_name=short_title,
                            index=False)
                fixXLColumns(
                    df,
                    self.excelWriter.sheets[short_title])
                self.excel_titles[short_title] = title
            else:
                title = ("LinePlot: HC " + dstr +
                         " to bearing " + sub_string)
                short_title = 'HC_' + dstrshort + '_line ' + sub_str
                blist_far = createPlotlyLinePlot(
                    df=df,
                    xrange=[-180, 180],
                    title="HC " + dstr +
                          " to bearing " + sub_string,
                    description=description,
                    include_plotly=(not self.plotly_included)
                )
                if not self.plotly_included:
                    self.plotly_included = True
                self.plotly_columns = self.plotly_columns + blist_far
                df.to_excel(self.excelWriter,
                            sheet_name='HC_' + dstrshort + '_line ' + sub_str,
                            index=False)
                fixXLColumns(
                    df,
                    self.excelWriter.sheets[
                        'HC_' + dstrshort + '_line ' + sub_str])
                self.excel_titles[short_title] = title

    def figure_boxplot_PREF(self, par, conditions):

        pref_array = []
        # for all conditions
        for condition_idx, condition in enumerate(conditions):

            # init
            pref = []
            pref_trials = [
                self.dict[condition].trial[index] for index in sorted(
                    np.unique(self.dict[condition].trial, return_index=True)[1])
            ]
            trial_numbers = [
                self.dict[condition].trial_number[index] for index in sorted(
                    np.unique(
                        self.dict[condition].trial_number,
                        return_index=True)[1])
            ]
            trial_dict = dict(list(zip(trial_numbers, pref_trials)))
            # full_condition = self.dict[condition].full_condition
            # for all trials
            for trial_number in trial_numbers:
                # init
                time_up = 0.
                time_down = 0.

                # select trial and not nan idx
                trial_idx = self.dict[condition].trial_number == trial_number
                not_nan_idx = ~np.isnan(self.dict[condition].spine4[:, 0])

                # spine4_y
                spine4_y = self.dict[condition].spine4[
                    not_nan_idx *
                    trial_idx,
                    1]

                # time up and down
                time_down = np.sum(spine4_y < 0.) * float(par['dt'])
                time_up = np.sum(spine4_y > 0.) * float(par['dt'])

                prefval = (time_up - time_down) / (time_down + time_up)
                pref.append(prefval.astype(float))
                pref_array.append([condition,
                                   trial_dict[trial_number],
                                   prefval.astype(float)])

            # convert to array
            pref = np.array(pref)

        # save data
        if par['save_data']:
            df = pandas.DataFrame(pref_array)
            df.sort_values(by=[0, 1], inplace=True)
            df.columns = ["Group", "Trial Name", "PREF"]
            df.to_excel(self.excelWriter,
                        sheet_name='PREF',
                        index=False)
            mf.fixXLColumns(
                df,
                self.excelWriter.sheets[
                    'PREF'])
            self.excel_titles['PREF'] = 'PREFs'
