from Individuals import Individuals
from all_tracks import All_tracks
import os
import sys
from Track import Track

def script_make_multiple_groups_figures(par):
    experiment = par['experiment_name']
#    for p in par:
#        print(p +": "+ str(par[p]))

    groups = par['groups']
    par['orig_groups'] = par['groups']
    if par['groups'] == ['all']:
        found_groups = []
        for x in os.listdir(par['basedir']):
            if os.path.isdir(os.path.join(par['basedir'], x)) and x != 'tmp':
                for d in os.listdir(os.path.join(par['basedir'], x)):
                    found_groups.append(os.path.join(x, d))
        groups = sorted(found_groups)
    groups = [x.replace('/', '_-_') for x in groups]

    # grouped_conditions =
    print('Executing multiple groups analysis for ' + str(groups))
    all_tracks = All_tracks(
        par,
        experiment,
        groups)

    # timeit
    # start_time = timeit.default_timer()

    for variable_name in ['HC_rate', 'run_speed','Abs_HC_angle']:
        for q in ['','q1','q2','q3','q4']:
            all_tracks.figure_boxplot_variable(
                par,
                conditions=groups,
                variable_name=variable_name,quadrant=q)

            if variable_name == 'HC_rate':
                all_tracks.figure_boxplot_variable(
                    par,
                    conditions=groups,
                    variable_name=variable_name,

                    subthreshold=True)



    """
    all_tracks.figure_boxplot_PREF(
        par,
        conditions=groups)

    # Time resolved preferences
    all_tracks.figure_time_resolved_PREFs(
        par,
        conditions=groups)

    all_tracks.figure_hist(par, conditions=groups, key='duration')

    #all_tracks.write_html("Prefs")
    limitation = 'nl'
    # for limitation in ['nl','l','il']:

    all_tracks.figure_HC_reorientation_boxplot(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        bearing_limited=limitation
    )

    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        bearing_limited=limitation
    )

    all_tracks.figure_boxplot_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name='HC_rate',
        towards_away_limited=limitation
    )

    all_tracks.figure_boxplot_variable_depending_on_bearing_distance_split(
        par,
        conditions=groups,
        variable_name='HC_rate',
        towards_away_limited=limitation,
        distance=par['d_split']
    )

    all_tracks.figure_boxplot_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name='HCs',
        towards_away_limited=limitation
    )
    all_tracks.figure_boxplot_variable_depending_on_bearing_distance_split(
        par,
        conditions=groups,
        variable_name='HCs',
        towards_away_limited=limitation,
        distance=par['d_split']
    )

    angleDependent = ['Abs_HC_angle_turn_TA',
                      'Abs_HC_angle_head_TA',
                      'HC_angle',
                      'HC_rate']

    histogramVars = ['Abs_HC_angle_head_TA',
                     'HC_angle']

    # for variable_name in ['Abs_HC_angle_turn_TA', 'Abs_HC_angle_head_TA']:

    all_tracks.figure_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name='HCs',
        subthreshold=False)

    all_tracks.figure_variable_depending_on_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        variable_name='HCs',
        subthreshold=False
    )

    all_tracks.figure_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name='Abs_HC_angle_head_TA',
        subthreshold=False)

    for variable_name in ['HC_rate', 'run_speed', 'HC_angle',
                          'Abs_HC_angle_head_TA',
                          'Abs_HC_angle_turn_TA']:

        # all_tracks.figure_variable_depending_on_bearing(
        #    par,
        #    conditions=groups,
        #   variable_name=variable_name)

        all_tracks.figure_boxplot_variable(
            par,
            conditions=groups,
            variable_name=variable_name)

        if variable_name in angleDependent:
            all_tracks.figure_boxplot_variable(
                par,
                conditions=groups,
                variable_name=variable_name,
                subthreshold=True)
            all_tracks.figure_boxplot_variable(
                par,
                conditions=groups,
                variable_name=variable_name,
                large_HC=0)
            all_tracks.figure_variable_depending_on_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                subthreshold=True)
            all_tracks.figure_variable_depending_on_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                large_HC=0)

        all_tracks.figure_heatmap_distance_and_bearing(
            par,
            conditions=groups,
            variable_name=variable_name)

        if variable_name in angleDependent:
            all_tracks.figure_heatmap_distance_and_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                subthreshold=True)
            all_tracks.figure_heatmap_distance_and_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                large_HC=0)

        all_tracks.figure_boxplot_variable_depending_on_bearing(
            par,
            conditions=groups,
            variable_name=variable_name)
        if variable_name in angleDependent:
            all_tracks.figure_boxplot_variable_depending_on_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                subthreshold=True)
            all_tracks.figure_boxplot_variable_depending_on_bearing(
                par,
                conditions=groups,
                variable_name=variable_name,
                large_HC=0)

        all_tracks.figure_variable_depending_on_bearing_distance_split(
            par,
            conditions=groups,
            variable_name=variable_name,
            distance=par['d_split'])
        if variable_name in angleDependent:
            all_tracks.figure_variable_depending_on_bearing_distance_split(
                par,
                conditions=groups,
                variable_name=variable_name,
                distance=par['d_split'],
                subthreshold=True)
            all_tracks.figure_variable_depending_on_bearing_distance_split(
                par,
                conditions=groups,
                variable_name=variable_name,
                distance=par['d_split'],
                large_HC=0)

        all_tracks.figure_boxplot_variable_distance_split(
            par,
            conditions=groups,
            variable_name=variable_name,
            distance=par['d_split'])
        if variable_name in angleDependent:
            all_tracks.figure_boxplot_variable_distance_split(
                par,
                conditions=groups,
                variable_name=variable_name,
                distance=par['d_split'],
                subthreshold=True)
            all_tracks.figure_boxplot_variable_distance_split(
                par,
                conditions=groups,
                variable_name=variable_name,
                distance=par['d_split'],
                large_HC=0)

        if variable_name in histogramVars:
            all_tracks.figure_hist(par, conditions=groups, key=variable_name)

        all_tracks.write_html(variable_name)

    # Proportion of HCs box
    all_tracks.figure_proportion_of_HCs_boxplot(
        par,
        conditions=groups)
    # Proportion of HCs box subthreshold
    all_tracks.figure_proportion_of_HCs_boxplot(
        par,
        conditions=groups,
        subthreshold=True)

    # Proportion of HCs box all HC angles
    all_tracks.figure_proportion_of_HCs_boxplot(
        par,
        conditions=groups,
        large_HC=0)

    # Proportion of HCs line all HC angles split distance
    all_tracks.figure_variable_depending_on_bearing_distance_split(
        par,
        conditions=groups,
        variable_name="HC Proportion",
        distance=par['d_split'])

    # Proportion of HCs Reorientation (positive/negative) split distance
    all_tracks.figure_variable_depending_on_bearing_distance_split(
        par,
        conditions=groups,
        variable_name="HC_Proportion_Reor",
        distance=par['d_split'])

    # Proportion of HCs line all HC angles
    all_tracks.figure_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name="HC Proportion")

    # Proportion of HCs Reorientation (positive/negative)
    all_tracks.figure_variable_depending_on_bearing(
        par,
        conditions=groups,
        variable_name="HC_Proportion_Reor")

    # Proportion of HCs box split by d_split box
    all_tracks.figure_proportion_of_HCs_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'])

    # Proportion of HCs box split by d_split box
    all_tracks.figure_proportion_of_HCs_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        subthreshold=True)

    # Proportion of HCs box split by d_split box all HC angles
    all_tracks.figure_proportion_of_HCs_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        large_HC=0)

    all_tracks.write_html("HC_Proportion")

    sub_mean = False

    # for sub_mean in [True,False]

    # HC Reorientation split by distance box
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        subtract_mean=sub_mean)

    # HC Reorientation split by distance box subthreshold
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        subthreshold=True,
        subtract_mean=sub_mean)

    # HC Reorientation split by distance box all HC angles
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        subthreshold=False,
        large_HC=0,
        subtract_mean=sub_mean)

    # HC Reorientation whole dish box
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True)

    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        mean_of_bins=True),

    # HC Reorientation whole dish box subthreshold
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        subthreshold=True,
        subtract_mean=sub_mean)

    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        heading='towards')

    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,

        heading='away')

    # HC Reorientation whole dish box all HC angles
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        large_HC=0,
        subtract_mean=sub_mean)

    # HC reorientation split by distance line
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        subtract_mean=sub_mean, )

    # HC reorientation split by distance line subthreshold
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        subthreshold=True,
        subtract_mean=sub_mean)

    # HC reorientation split by distance line all HC angles
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=True,
        large_HC=0,
        subtract_mean=sub_mean)

    # HC Reorientation whole dish line
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True)

    # HC Reorientation whole dish line subthreshold
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        subthreshold=True,
        subtract_mean=sub_mean)

    # HC Reorientation whole dish line all HC angles
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=True,
        large_HC=0,
        subtract_mean=sub_mean)

    all_tracks.write_html("HC_Reorientation")

    # HC Accuracy split by distance box
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False)

    # HC Accuracy split by distance box subthreshold
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False,
        subthreshold=True)

    # HC Accuracy split by distance box all angles
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False,
        large_HC=0)

    # HC Accuracy whole dish box
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False)

    # HC Accuracy whole dish box subthreshold
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False,
        subthreshold=True)

    # HC Accuracy whole dish box all angles
    all_tracks.figure_HC_reorientation_boxplot_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False,
        large_HC=0)

    # HC reorientation split by distance line
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False)

    # HC reorientation split by distance line subthreshold
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False,
        subthreshold=True)

    # HC reorientation split by distance line all angles
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=par['d_split'],
        reorientation=False,
        large_HC=0)

    # HC Accuracy whole dish line
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False)

    # HC Accuracy whole dish line subthreshold
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False,
        subthreshold=True)

    # HC Accuracy whole dish line all angles
    all_tracks.figure_HC_reorientation_to_bearing_distance_split(
        par,
        conditions=groups,
        distance=0,
        reorientation=False,
        large_HC=0)

    all_tracks.write_html("HC_Accuracy")

    all_tracks.makeExcelTOC()
    # all_tracks.figure_run_speed_boxplot(
    #     par,
    #     conditions=groups)
    """
    all_tracks.excelWriter.save()

import time
def script_make_individual_figures(par):
    experiment = par['experiment_name']
    groups = par['groups']
    par['orig_groups'] = par['groups']
    if par['groups'] == ['all']:
        found_groups = []
        for x in os.listdir(par['basedir']):
            if os.path.isdir(os.path.join(par['basedir'], x)) and x != 'tmp':
                for d in os.listdir(os.path.join(par['basedir'], x)):
                    found_groups.append(os.path.join(x, d))
        groups = sorted(found_groups)
    groups = [x.replace('/', '_-_') for x in groups]

    # grouped_conditions =
    print('Executing group analysis of individual larvae for ' + str(groups), flush=True)

    tracks = Individuals(
        par,
        experiment=experiment,
        grouped_conditions=groups)

    tracks.figure_proportion_of_time_boxplot(par, groups)
    tracks.figure_proportion_of_time_boxplot_distance_split(
        par, groups,
        distance=par['d_split'])

    # tracks.individuals_plot(par,groups,limit=15)

  
    for param in ['INS_reorient', 'HC_reorientation', 'HC_accuracy', 'HC_rate', 'run_speed',
                  ]:

        # Plot total values only in the first case
        plot_total = True
        for mod in ['bearing', 'distance', 'time']:
            tracks.figure_boxplot_variable_depending_on_parameter(
                par,
                groups,
                param,
                parameter=mod,
                plot_total=plot_total)

            if plot_total:
                plot_total = False

            tracks.figure_boxplot_variable_depending_on_parameter(
                par,
                groups,
                param,
                subthreshold=True,
                parameter=mod,
                plot_total=False)

        
    tracks.figure_proportion_of_time_boxplot(par, groups)
    tracks.figure_proportion_of_time_boxplot_distance_split(
        par, groups,
        distance=par['d_split'])

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_turn_TA')

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_turn_TA',
        subthreshold=True)

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_turn_TA',
        subthreshold=False,
        large_HC=0)

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'HC_angle')

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'HC_angle',
        subthreshold=True)

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'HC_angle',
        subthreshold=False,
        large_HC=0)

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_head_TA')

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_head_TA',
        subthreshold=True)

    tracks.figure_boxplot_variable_depending_on_parameter(
        par,
        groups,
        'Abs_HC_angle_head_TA',
        subthreshold=False,
        large_HC=0)

    tracks.makeExcelTOC()
    tracks.excelWriter.save()



def script_example_analysis_of_single_track(par):
    # here one can play arround to understand the analysis of a single track

    # for debugging
    global track

    # parameters
    # par = define_parameters()
    par['dt'] = 1. / float(par['fps'])
    # par['save_figure'] = False

    # select single track
    experiment = par['experiment_name']
    condition = par['condition']
    trial = par['trial_name']
    track_number = par['track_name'] 
    print(experiment)
    print(trial)
    print(track_number)
    # init track
    track = Track(par, experiment=experiment,
                  condition=condition,
                  trial=trial, track_number=track_number)

    # cut track within valid range (optional)
    # print 'track start = %.2f, track end = %.2f' % (
    #    track.time[0], track.time[-1])
    # if True:
    #    track.cut_track_before_analysis(par, t_start = 30, t_end = 45)

    # analyze and save track
    track.compute_time_series_variables(par)
    track.detect_steps(par)
    track.detect_HCs(par)
    track.compute_scalar_variables(par)
    track.check_analyzed_track(par)
    track.save(par)

    track_string_name = (str(experiment) + '_' +
                         str(condition) + '_' + str(track_number))

    # figure track on dish
    if 'track_on_dish' in par['single_track_analysis']:
        track.figure_track_on_dish(par, keys=['steps', 'HCs', 'start', 'end'],
                                   figure_name='track_on_dish_' +
                                               track_string_name)
        track.figure_track_on_dish_whole(
            par, keys=['start'],
            figure_name='track_on_dish_whole_' +
                        track_string_name)

    # animation
    if 'animate_track' in par['single_track_analysis']:
        # track.animate_track(
        #     par,
        #     speed=1,
        #     zoom_dx=6,
        #     save_movie=True,
        #     movie_name='STEP_complete_track_' +
        #     track_string_name)
        track.animate_track_and_vars(
            par,
            speed=1,
            zoom_dx=6,
            save_movie=True,
            movie_name='complete_track_' +
                       track_string_name)

    # figure time-series variables
    if 'time_series_separate' in par['single_track_analysis']:
        track.figure_time_series_variables(
            par,
            keys=[
                'all_variables',
                # 'steps',
                'HCs'],
            figure_name='time_series_variables_' + track_string_name)
        track.figure_time_series_variables_sep(
            par,
            keys=[
                'all_variables',
                # 'steps',
                'HCs'],
            figure_name='time_series_' + track_string_name)

    # figure time-series variables all in one for phase relation
    if 'time_series_all_in_one' in par['single_track_analysis']:
        track.figure_time_series_variables_all_in_one(
            par,
            figure_name='time_series_variables_all_' + track_string_name)

    # init tracks with a single track, in order to use tracks figures
    if False:
        tracks = Tracks(par, experiment=experiment,
                        condition=condition, which_trials=[trial],
                        which_tracks=[track_number])

    # histogram of inter-step-intervals
    if False:
        key = 'INS_interval'
        tracks.figure_hist(par, key=key, figure_name='hist_long_example_track')

    # scatter plots
    if False:
        keys = [
            'INS_interval_vs_INS_distance',
            'tail_speed_forward_vs_head_speed_forward',
            'tail_angular_speed_vs_n_steps_in_HC',
            'tail_angular_speed_vs_HC_angle',
            'step_HC_interval_vs_HC_step_interval',
            'bending_angle_vs_INS_turn']
        for key in keys:
            tracks.figure_scatter_plot_with_hist(
                par,
                key=key,
                figure_name=key +
                '_' +
                condition)




def script_simulate_single_track():
    # for debugging
    global track_sim

    # define parameters
    par = define_parameters()
    par['dt'] = 1. / float(par['fps'])
    par['t_end'] = 200

    track_sim = Track_sim(par)

    track_sim.simulate(par)

    track_sim.append_spine_and_contour(par)

    track_sim.filter_track_sim(par)

    track_sim.compute_time_series_variables(par)

    track_sim.animate_track(par, speed=1, zoom_dx=6,
                            save_movie=False, movie_name='track_sim')

