
from Scripts import *
from Tracks import *
import sys


import argparse





def define_parameters(parsed_args=None):
	par = {
	'figure_dir': parsed_args.figure_folder[0],
	'data_dir': parsed_args.data_folder[0],
	'save_data': parsed_args.save_data,
	'save_figure': parsed_args.save_figure,
	'movie_dir': parsed_args.movie_folder[0],
	'duration': parsed_args.duration,
	'start_time': parsed_args.time_range_start,
	'end_time': parsed_args.time_range_end,
	'large_HC': 20.,
	'area_line_width': 7,
	# general
	'fps': parsed_args.fps[0],
	'fig_width': 8. / 2.54,  # 8cm is typical one-column figure width
	# gap between hc and run in frames
	'gap': 24,
	# minimal track duration when writing database
	'minimal_duration': parsed_args.minimal_duration[0],
	# individual analysis filters
	'individual_range_begin': parsed_args.begin_range,
	'individual_range_end': parsed_args.end_range,
	# 'individual_duration_min': parsed_args.min_valid_duration,
	'individual_duration_min': 0,
	# number of filter points for track init, default is 2 for barcelona
	'n_filter_points': 2,  # (0,1,2)
	# Range to consider towards (symmetric around 0)
	'to_range': np.pi/2.,
	'away_range': np.pi/2.,

	'radius_dish': parsed_args.radius[0],  # radius of petri dish in mm,

	# animation dpi
	'animation_dpi': 150,
	# force rebuild pickles
	'rebuild': False,
	# sample interval
	'dt': 1. / parsed_args.fps[0],

	# distance to split groups for distance plots
	'd_split': int(2 * parsed_args.radius[0] / 2.316),
	# 'd_split': 40,

	# threshold on tail_speed_forward for step detection
	'threshold_tail_speed_forward': 0.6,

	# HC detection parameters
	'threshold_head_vector_angular_speed': np.deg2rad(35.0),
	'threshold_back_vector_angular_speed': np.deg2rad(45.0),
	'threshold_n_steps_per_HC': 1,  # 1 step per HC is allowed

	# heatmap analysis
	'n bins': 60,
	'edges_bearing': np.deg2rad(np.linspace(-180, 180, 60 + 1)),
	# 'edges_distance': np.linspace(0, 150, 60 + 1),
	# 'edges_distance': np.linspace(0, 84, 60 + 1),
	'edges_distance': np.linspace(0, 2 * parsed_args.radius[0], 60 + 1),

	# simulation parameters

	# motion parameters
	'angular_speed': np.deg2rad(73.3),
	'mean_INS_interval': 0.7,
	'mean_INS_distance': 0.7,
	'spine_length': 3.8,

	'p(HC|step)': 0.1,
	'p(HC|HC)': 0.4,

	'HC_angles_sigma': np.deg2rad(46.3),
	# PREF limit for bimodel indiviual cases
	'preference_limit': 0.9,
	# Individual limits:
	'ind_lim_INS_distance': 0,
	'ind_lim_INS_interval': 0,
	'ind_lim_INS_turn': 0,
	'ind_lim_INS_reorient': 0,
	'ind_lim_HC_rate': 0,
	'ind_lim_HC_reorientation': 0,
	'ind_lim_HC_accuracy': 0,
	'ind_lim_run_speed': 0,
	'ind_lim_HC_angle': 0,
	'ind_lim_Abs_HC_angle_turn_TA': 0,
	'ind_lim_Abs_HC_angle_head_TA': 0,
	'ind_lim_prop': 0,

	# time
	# 't_start': 0.,
	# TODO: Change this according to movie
	# 't_end': 180.
	}
	if isinstance(par['individual_range_begin'], list):
		par['individual_range_begin'] = par['individual_range_begin'][0]

	if isinstance(par['individual_range_end'], list):
		par['individual_range_end'] = par['individual_range_end'][0]

	if isinstance(par['individual_duration_min'], list):
		par['individual_duration_min'] = par['individual_duration_min'][0]

	if par['individual_range_end'] == -1:
		par['individual_range_end'] = par['duration']

	if parsed_args.odor_A == '':
		par['odor_A'] = None
	else:
		par['odor_A'] = np.array(list(map(float, parsed_args.odor_A[0].split(','))))
	# print 'odor_A from arg read as: ' + str(par['odor_A'])
	if parsed_args.odor_B == '':
		par['odor_B'] = None
	else:
		par['odor_B'] = np.array(list(map(float, parsed_args.odor_B[0].split(','))))

	if isinstance(par['duration'], list):
		par['duration'] = par['duration'][0]

	if isinstance(par['start_time'], list):
		par['start_time'] = par['start_time'][0]

	if isinstance(par['end_time'], list):
		par['end_time'] = par['end_time'][0]

	if par['end_time'] == -1:
		par['end_time'] = par['duration']

	par['duration'] = int(par['duration'])
	par['start_time'] = int(par['start_time'])
	par['end_time'] = int(par['end_time'])

	par['time_restrict'] = True
	if par['start_time'] == 0 and par['end_time'] == par['duration']:
		par['time_restrict'] = False

	par['path_suffix'] = '.pkl'
	if par['time_restrict'] is True:
		par['path_suffix'] = ('T' + str(par['start_time']) +
						  '-' + str(par['end_time']) + '.pkl')

		print("Time S - E: " + str(par['start_time']) + ' - ' + str(par['end_time']))
		print("Duration: " + str(par['duration']))
		print("Time restrict: " + str(par['time_restrict']))

	if parsed_args.analysis_mode[0] == 'single':

		if parsed_args.track_file[0] is not None:
			par['track_file'] = parsed_args.track_file[0]
			path_list = par['track_file'].split('/')

			par['track_name'] = path_list[-1].split('.')[0]
			par['trial_name'] = path_list[-2]
			par['condition'] = path_list[-3]
			par['group_name'] = path_list[-4]
			par['tracked_on'] = path_list[-5]
			par['experiment_name'] = path_list[-6]
			par['parent_dir'] = '/'.join(path_list[:-6])
		else:
			par['parent_dir'] = parsed_args.basedir[0]
			par['figure_dir'] = parsed_args.figure_folder[0]
			par['data_dir'] = parsed_args.data_folder[0]
			par['movie_dir'] = parsed_args.movie_folder[0]
			par['experiment_name'] = parsed_args.experiment_name[0]
			par['group_name'] = parsed_args.group_name[0]
			par['condition'] = parsed_args.condition_name[0]
			par['trial_name'] = parsed_args.trial_name[0]
			par['track_name'] = parsed_args.track_name[0]

		par['single_track_analysis'] = parsed_args.single_track_analysis
		if 'all' in par['single_track_analysis']:
			par['single_track_analysis'] = ['track_on_dish', 'animate_track',
											'single_contour',
											'time_series_separate',
											'time_series_all_in_one']

	elif parsed_args.analysis_mode[0] == 'trial':
		par['trial_analysis'] = parsed_args.trial_analysis
		if 'all' in par['trial_analysis']:
			par['trial_analysis'] = [
				'tracks_on_dish', 'tracks_on_dish_heatmap',
				'density_on_distance_time',
				'time_resolved_PREF', 'histograms',
				'scatter_plot_with_hist',
				'3d_scatter_plot',
				'heatmap_distance_bearing_projected',
				'heatmap_distance_bearing',
				'all']

		if parsed_args.trial_dir[0] is not None:
			par['trial_dir'] = parsed_args.trial_dir[0]
			path_list = par['trial_dir'].split('/')
			par['trial_name'] = path_list[-1]
			par['condition'] = path_list[-2]
			par['group_name'] = path_list[-3]
			par['tracked_on'] = path_list[-4]
			par['experiment_name'] = path_list[-5]
			par['parent_dir'] = '/'.join(path_list[:-5])

	elif parsed_args.analysis_mode[0] == 'group':
		par['group_analysis'] = parsed_args.group_analysis
		if 'all' in par['group_analysis']:
			par['group_analysis'] = [
				'tracks_on_dish', 'tracks_on_dish_heatmap',
				'track_numbers_per_time',
				'density_on_distance_time',
				'time_resolved_PREF', 'histograms',
				'scatter_plot_with_hist',
				'3d_scatter_plot',
				'heatmap_distance_bearing_projected',
				'heatmap_distance_bearing',
				'HC_direction_serial_correlation',
				'all']

		if parsed_args.group_dir[0] is not None:
			par['group_dir'] = parsed_args.group_dir[0]
			path_list = par['group_dir'].split('/')
			par['condition'] = path_list[-1]
			par['group_name'] = path_list[-2]
			par['tracked_on'] = path_list[-3]
			par['experiment_name'] = path_list[-4]
			par['parent_dir'] = '/'.join(path_list[:-4])

	elif parsed_args.analysis_mode[0] == 'multiple-groups':
		par['multiple_groups_analysis'] = parsed_args.multiple_groups_analysis
		if 'all' in par['multiple_groups_analysis']:
			par['multiple_groups_analysis'] = [
				'variable_depending_on_bearing',
				'boxplot_variable_depending_on_bearing',
				'time_resolved_PREFs',
				'all'
			]

		if parsed_args.basedir[0] is not None:
			par['basedir'] = parsed_args.basedir[0]
			path_list = par['basedir'].split('/')
			par['tracked_on'] = path_list[-1]
			par['experiment_name'] = path_list[-2]
			par['parent_dir'] = '/'.join(path_list[:-2])
			par['groups'] = parsed_args.groups
			print(par['groups'])
			print(par['parent_dir'])

	elif parsed_args.analysis_mode[0] == 'individual':
		par['individual_analysis'] = parsed_args.individual_analysis
		if 'all' in par['individual_analysis']:
			par['individual_analysis'] = [
				'tracks_on_dish', 'tracks_on_dish_heatmap',
				'track_numbers_per_time',
				'density_on_distance_time',
				'time_resolved_PREF', 'histograms',
				'scatter_plot_with_hist',
				'3d_scatter_plot',
				'heatmap_distance_bearing_projected',
				'heatmap_distance_bearing',
				'all']

		if parsed_args.basedir[0] is not None:
			par['basedir'] = parsed_args.basedir[0]
			path_list = par['basedir'].split('/')
			par['tracked_on'] = path_list[-1]
			par['experiment_name'] = path_list[-2]
			par['parent_dir'] = '/'.join(path_list[:-2])
			par['groups'] = parsed_args.groups
			print(par['groups'])
			print(par['parent_dir'])
	return par

# ==============================================================================
# Parse function
# ==============================================================================

def parse_commandline(argv):
	global basedir

	# Parsing command line to decide on Run sequence:
	parser = argparse.ArgumentParser(
	description="Statistical Analyses for larvae experiments.")
	required_args_parser = parser.add_argument_group('Required Arguments')
	general_args_parser = parser.add_argument_group('General Arguments')

	single_track_parser = parser.add_argument_group(
	'Single Track Mode',
	'Analysis of a given single track. Use either with\
	direct file path (--track-file) or by providing the\
	basedir experiment group trial and track information.\
	The track-file argument supersedes the basedir,experiment etc.\
	information.')
	trial_parser = parser.add_argument_group(
	'Trial Mode',
	'Analysis of a given trial (i.e. all tracks on a single petri dish).\
	Use either with direct file path (--trial-dir) or by providing the\
	basedir experiment group and trial names.\
	The trial-dir argument supersedes the basedir,experiment etc.\
	information.')

	group_parser = parser.add_argument_group(
	'Group Mode',
	'Analysis of a given experimental group.\
	Use either with direct file path (--group-dir) or by providing the\
	basedir, experiment and group names.\
	The group-dir argument supersedes the basedir,experiment etc.\
	information.')

	multiple_group_parser = parser.add_argument_group(
	'Multiple Group Mode',
	'Analysis of a given experimental group.\
	by providing the\
	basedir, experiment and group names.\
	The group-dir argument supersedes the basedir,experiment etc.\
	information.')

	individual_parser = parser.add_argument_group(
	'Individual Mode',
	'Analysis of individuals within a given experimental group.\
	Use either with direct file path (--group-dir) or by providing the\
	basedir, experiment and group names.\
	The group-dir argument supersedes the basedir,experiment etc.\
	information.')

	required_args_parser.add_argument(
	'--basedir',
	nargs=1,
	help='Name of the top level directory where the\
	experiment folder resides.')

	required_args_parser.add_argument(
	'--experiment-name',
	default=' ',
	nargs=1,
	help='Name of the experiment.')

	required_args_parser.add_argument(
	'--analysis-mode',
	nargs=1,
	required=True,
	choices=['single', 'trial', 'group', 'multiple-groups', 'individual'],
	help='Mode of analysis (required).')

	general_args_parser.add_argument(
	'--fps',
	nargs=1,
	default=[16.0],
	type=float,
	help='fps used for filming the larvae.')

	general_args_parser.add_argument(
	'--minimal-duration',
	nargs=1,
	default=[3.0],
	type=float,
	help='consider only tracks larger than the value given (in seconds).')

	general_args_parser.add_argument(
	'--radius',
	nargs=1,
	default=[44.0],
	type=float,
	help='Radius of the petri dish in (mm).')
	general_args_parser.add_argument(
	'--time-range-start',
	nargs=1,
	default=0,
	help='Restrict analysis to data collected after N sec. (Default: 0)')
	general_args_parser.add_argument(
	'--time-range-end',
	nargs=1,
	default=-1,
	help='Restrict analysis to data collected before N sec. (Default: -1)')
	general_args_parser.add_argument(
	'--duration',
	nargs=1,
	default=180,
	help='Duration of each experiment in seconds. (Default: 180)')
	general_args_parser.add_argument(
	'--odor-A',
	nargs=1,
	default='',
	help='Position of odor A using x,y coords \
	(--paired-odor 0.0,44.2)')

	general_args_parser.add_argument(
	'--odor-B',
	nargs=1,
	default='',
	help='Position of odor B using x,y coords \
	(--unpaired-odor 0.0,44.2)')

	general_args_parser.add_argument(
	'--save-figure', action='store_true',
	help='Save figures to figure folder.')

	general_args_parser.add_argument(
	'--no-save-figure', action='store_false',
	help='Do not save figures.')

	general_args_parser.add_argument(
	'--figure-folder',
	nargs=1,
	default='.',
	help='Folder into which to save the figures.')

	general_args_parser.add_argument(
	'--save-data', action='store_true',
	help='Save data to data folder.')

	general_args_parser.add_argument(
	'--no-save-data', action='store_false',
	help='Do not save data.')

	general_args_parser.add_argument(
	'--data-folder',
	nargs=1,
	default='.',
	help='Folder into which to save the data.')

	general_args_parser.add_argument(
	'--movie-folder',
	nargs=1,
	default='.',
	help='Folder into which to save the movies.')

	single_track_parser.add_argument(
	'--animate-track',
	action='store_true',
	help='Display an animated\
	larva moving around the dish.')

	single_track_parser.add_argument(
	'--track-file',
	nargs=1,
	help='Path for CSV file containing track info.')

	single_track_parser.add_argument(
	'--group-name',
	default=' ',
	nargs=1,
	help='Name of group (folder name).')

	single_track_parser.add_argument(
	'--condition-name',
	default=' ',
	nargs=1,
	help='Name of group (folder name).')

	single_track_parser.add_argument(
	'--trial-name',
	nargs=1,
	help='Name of trial (folder name).')

	single_track_parser.add_argument(
	'--track-name',
	nargs=1,
	help='File name of track (csv file).')

	single_track_parser.add_argument(
	'--single-track-analysis',
	nargs='+',
	choices=['track_on_dish', 'animate_track',
			 'single_contour', 'time_series_separate',
			 'time_series_all_in_one', 'all'],
	help='Kinds of analyses to perform.')

	trial_parser.add_argument(
	'--trial-dir',
	nargs=1,
	help='Path for CSV file containing track info.')

	group_parser.add_argument(
	'--group-dir',
	nargs=1,
	help='Path for the folder containing group data.')

	group_parser.add_argument(
	'--condition-dir',
	nargs=1,
	help='Path for a single reciprocal from the group')

	trial_parser.add_argument(
	'--trial-analysis',
	nargs='+',
	choices=['tracks_on_dish'],
	help='Kinds of analyses to perform.')

	group_parser.add_argument(
	'--group-analysis',
	nargs='+',
	choices=['tracks_on_dish', 'density_on_distance_time',
			 'time_resolved_PREF', 'histograms',
			 'scatter_plot_with_hist',
			 '3d_scatter_plot',
			 'heatmap_distance_bearing_projected',
			 'heatmap_distance_bearing',
			 'HC_direction_serial_correlation',
			 'all'],
	help='Kinds of analyses to perform.')

	multiple_group_parser.add_argument(
	'--groups',
	nargs='+',
	help='Groups to plot next to each other.')

	multiple_group_parser.add_argument(
	'--multiple-groups-analysis',
	nargs='+',
	choices=['variable_depending_on_bearing',
			 'boxplot_variable_depending_on_bearing',
			 'time_resolved_PREFs',
			 'all'],
	help='Kinds of analyses to perform.')

	individual_parser.add_argument(
	'--begin-range',
	nargs=1,
	default=0,
	help='Consider tracks starting after the given value')

	individual_parser.add_argument(
	'--end-range',
	nargs=1,
	default=-1,
	help='Consider tracks starting after the given value')

	individual_parser.add_argument(
	'--min-valid-duration',
	nargs=1,
	help='Consider tracks whose valid duration is more than the\
	 given value. We consider only stretches of data longer than the\
	 minimal_duration parameter')

	individual_parser.add_argument(
	'--individual-analysis',
	nargs='+',
	choices=['all'],
	help='Kinds of analyses to perform.')

	args = parser.parse_args(argv[1:])
	par = define_parameters(args)
	if args.analysis_mode[0] == 'single':
		script_example_analysis_of_single_track(par)

	if args.analysis_mode[0] == 'group':
		script_make_group_figures(par)

	if args.analysis_mode[0] == 'multiple-groups':
		script_make_multiple_groups_figures(par)

	if args.analysis_mode[0] == 'individual':
		script_make_individual_figures(par)




# ==============================================================================
# Notes
# =============================================================================

# Before you start, set up the following folder structure:
# some_parent_dir/data
# some_parent_dir/data/tmp
# some_parent_dir/figures
# some_parent_dir/movies
# some_parent_dir/src

# the experimental is organized in the folder structure:
# some_parent_dir/data/experiment/condition/trial


# ==============================================================================
# Run section
# ==============================================================================

# V1=np.array([44.0,0.0])
# V2=np.array([0.0,44.0])
# print clockwise_angle_from_first_to_second_vector(V1,V2)

parse_commandline(sys.argv)

# script_try_out_some_stuff()

#script_example_analysis_of_single_track()

# script_make_example_movies_for_manuscript()

# script_write_database_and_save_tracks_for_manuscript()

#script_make_all_manuscript_figures()

# script_timo_buzz_analysis()

# script_simulate_single_track() # work in progress
