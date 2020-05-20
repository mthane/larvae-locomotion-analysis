#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export DISPLAY=:0.0
PROGDIR=$(dirname $0)
if [ -z "$PROGDIR" ]; then
  $PROGDIR='.'
fi

BASEDIR_CONF="$1"
if [ -z "$BASEDIR_CONF" ]; then
  echo "Please provide config file for experiment."
  exit 1
fi

BASEDIR="$(dirname "$BASEDIR_CONF")"

if [ ! -d "$BASEDIR" ]; then
  echo "Folder $BASEDIR doesn't exist"
  exit 1
else
  if [ ! -f "$BASEDIR_CONF" ]; then
    echo "No config file found for tracking please provide one (example follows) "
    echo "---- Example contents ---"
    echo 'EXPERIMENT_NAME="My experiment" \# A custom name for experiment otherwise folder name will be used.'
    echo 'RADIUS="41.5" \#for 9 cm petri dishes or '
    echo 'RADIUS="69" \#for 15 cm petri dishes '
    echo 'DURATION=180 \# total duration of the experiment'
    echo 'START_TIME=0 \# start from Nth second' 
    echo 'END_TIME=180 \# end at Nth second'
    echo '------ End -----'
    exit 1
  else
    source "$BASEDIR_CONF"
    if [ -z "$EXPERIMENT_NAME" ]; then
	EXPNAME="$(basename "$(dirname "$BASEDIR")")"
    else
	EXPNAME="$EXPERIMENT_NAME"
    fi

    if [ "$RADIUS" != "41.5" ] && [ "$RADIUS" != "69" ] && [ "$RADIUS" != "69.0" ]; then
      if [ -z "$RADIUS" ]; then
        echo "Warning: RADIUS is not provided, assuming 41.5"
        RADIUS="41.5"
      else
        echo "Warning!: RADIUS is $RADIUS"
      fi
    fi
    if [ -z "$DURATION" ] ; then
      echo "Warning!: No DURATION set. Assuming 180"
      DURATION=180
    fi
    if [ -z "$START_TIME" ]; then
      echo "Warning!: No START_TIME provided, assuming 0"
      START_TIME=0
    fi
    if [ -z "$END_TIME" ]; then
      echo "Warning!: No START_TIME provided, assuming DURATION"
      END_TIME="$DURATION"
    fi
  fi
fi

shift

#BASEDIR="/media/epaisios/S2/58E02_dark_light/Tracked_09_2015_pcr7280_22"
#BASEDIR="/media/epaisios/c4ad2a4a-feb1-4f43-b3a6-37370c541eb2/Joerg/Exp_11/Tracked_on_28092015"
#BASEDIR="/Processing/Joerg/Exp_11/TRACKED_ON_06_11_2015"
#BASEDIR="/Processing/Michael/WT_Aversive_appetitive/TRACKED_ON_13_01_2016M"
#BASEDIR="/Processing/Michael/WT_Aversive_appetitive/TRACKED_ON_13_01_2016MS"
#BASEDIR="/Processing/Michael/Vignesh Syn-CS/TRACKED_ON_08_04_2016"
#BASEDIR="/Processing/Michael/Vignesh Syn-CS/TRACKED_ON_22_04_2016"
#BASEDIR="/Processing/Michael/WT_Aversive_appetitive/TRACKED_ON_15_01_2016"
#BASEDIR="/Users/epaisios/sources/Analysis/WT_Aversive_appetitive/TRACKED_ON_15_01_2016"
#BASEDIR="/Users/epaisios/sources/Analysis/WT_Aversive_appetitive-new/TRACKED_ON_12_12_2016"
#BASEDIR="/Users/epaisios/sources/Analysis/Tessa_123_trials/TRACKED_ON_06_12_2016"
#BASEDIR="/Processing/Manos-Testing/WT_Aversive_appetitive-comp/TRACKED_ON_11_05_2016M"
#BASEDIR="/Processing/Michael/SS00864_p_f_a_ls/TRACKED_ON_31_03_2016M"
#BASEDIR="/Processing/Michael/SS00864_p_f_ar_so/TRACKED_ON_01_04_2016M"
#BASEDIR="/Processing/Michael/SS00864_combined/TRACKED_ON_31_03_2016M"
#BASEDIR="/Processing/Timo/MBE1s_Substitution_SS0864_1696_1757_Test/TRACKED_ON_14_04_2016"
#BASEDIR="/Processing/Michael/Stan_Fru_High_Low/TRACKED_ON_18_05_2016"
#BASEDIR="/Processing/Michael/Vignesh_Fru_High_Med_Low/TRACKED_ON_08_07_2016"
#BASEDIR="/Processing/Michael/Vignesh_Fru_High_Med_Low/TRACKED_ON_26_07_2016"
#BASEDIR="/Processing/Michael/Vignesh Syn-CS/TRACKED_ON_27_08_2016"
#BASEDIR="/Processing/Michael/Tessa_AM_Low_Med_High/TRACKED_ON_05_09_2016"
#BASEDIR="/Processing/Joerg/Exp_11/TRACKED_ON_31_08_2016"
#BASEDIR="/Processing/Michael/Vignesh_naive_AM_Low_Med_High/TRACKED_ON_22_10_2016"
RESULTSDIR="$(dirname "$BASEDIR")"
#EXPNAME="WT_With_MODEL_STATS"
#EXPNAME="WT_Aversive_appetitive-comp"
#EXPNAME="WT_Aversive_appetitive-Revision-new"
#EXPNAME="WT_Aversive_appetitive-Revision"
#EXPNAME="Tessa_123_trials"
#EXPNAME="WT_Aversive_appetitive-Revision-Subthreshold"
#EXPNAME="WT_Aversive_appetitive-Revision-Total"
#EXPNAME="SS00864_combined"
#EXPNAME="SS00864_p_f_a_ls"
#EXPNAME="SS00864_p_f_ar_so"
#EXPNAME="Exp_11"
#EXPNAME="Vignesh_Syn-CS"
#EXPNAME="MBE1s"
#EXPNAME="Stan_Fru_HL"
#EXPNAME=Vignesh_Fru_High_Med_Low
#EXPNAME="Tessa_AM_Low_Med_High"
#EXPNAME="Vignesh_naive"

#RADIUS="41.5"
#RADIUS="69"

#START_TIME=0
#END_TIME=180
#DURATION=180

IND_DURATION=30
IND_START=0
IND_END=180

if [ "$END_TIME" != "$DURATION" ] || [ "$START_TIME" != "0" ]; then

	F_SUFFIX="Analysis_${START_TIME}-${END_TIME}sec"
else
	F_SUFFIX="Analysis"
fi

RES_BASE="${RESULTSDIR}/${EXPNAME}-$F_SUFFIX/graphs"
DATA_BASE="${RESULTSDIR}/${EXPNAME}-$F_SUFFIX/data"
#SAVE_FIGURES="--no-save-figure" # DO NOT SAVE FIGURES

SAVE_FIGURES="--save-figure" # SAVE FIGURES
#SAVE_DATA="--no-save-data" # DO NOT SAVE DATA
SAVE_DATA="--save-data" # SAVE DATA

# ANALYSIS TYPES
# GROUP_ANALYSIS="all"
#echo START_TIME=$START_TIME
#echo END_TIME=$END_TIME
#echo DURATION=$DURATION
#echo EXPNAME=$EXPNAME
#echo BASEDIR=$BASEDIR
#echo BASEDIR_CONF=$BASEDIR_CONF
#echo RES_BASE=$RES_BASE
#echo DATA_BASE=$DATA_BASE


#exit 0

if [ -z "$1" ]; then
  echo "Need multiple/individual or group names to perform analysis"
fi

if [ "$1" = "merge" ] ; then
	shift
	for last; do true; done
	MERGENAME=$last
	FOLDER=$(echo $MERGENAME | sed 's/\//-/')
	if [ ! -d "$RES_BASE/$FOLDER" ]; then
		echo "Merged folder $FOLDER doesn't exist. Creating"
		mkdir -p "$RES_BASE/$FOLDER"
	fi
	if [ ! -d "$DATA_BASE/" ]; then
		echo "$DATA_BASE folder doesn't exist. Creating"
		mkdir -p "$DATA_BASE"
	fi
	OPWD="$PWD"
	while [ "$#" != "1" ]; do
	        CURRENTNAME=$1
		echo "Linking files from $CURRENTNAME to merged $MERGENAME"
		cd "$BASEDIR"
		mkdir -p "$MERGENAME"
		cd "$MERGENAME"
		ln -s "../../$CURRENTNAME/"* .
		shift
	done
	cd "$OPWD"
	python3 "$PROGDIR"/lin-analysis.py --duration $DURATION --time-range-start $START_TIME --time-range-end $END_TIME  --radius $RADIUS $SAVE_FIGURES $SAVE_DATA --analysis-mode=group --group-analysis all --group-dir "$BASEDIR"/"$MERGENAME" --data-folder=$DATA_BASE/ --figure-folder=$RES_BASE/$FOLDER --movie-folder=$RES_BASE/$FOLDER
 exit
fi

if [ "$1" = "multiple" ] ; then
	shift
        if [ -z "$1" ] ; then 
          echo "Need group names to perform multiple group analysis."
          exit 1
        fi
	GROUPDIR="$(echo $* | tr ' ' '_' | tr '/' '-')"
	if [ ! -d "$RES_BASE/multiple/$GROUPDIR" ]; then
		echo "Mulitple folder doesn't exist. Creating"
		mkdir -p "$RES_BASE/multiple/$GROUPDIR"
                if [ ! -d "$DATA_BASE" ]; then
                        echo "$DATA_BASE folder doesn't exist. Creating"
                        mkdir -p "$DATA_BASE"
                fi
        fi
        #if [ ! -e "$RES_BASE/multiple/$GROUPDIR/jquery.min.js" ] ; then
        cp $PROGDIR/resources/jquery.min.js "$RES_BASE/multiple/$GROUPDIR"
        #fi
	python3 "$PROGDIR"/lin-analysis.py --duration "$DURATION" --time-range-start "$START_TIME" --time-range-end "$END_TIME" --basedir "$BASEDIR" "$SAVE_FIGURES" "$SAVE_DATA" --radius "$RADIUS" --analysis-mode=multiple-groups --data-folder="$DATA_BASE"/ --multiple-groups-analysis all --groups $* --figure-folder="$RES_BASE"/multiple/"$GROUPDIR"/ --movie-folder="$RES_BASE"/multiple/"$GROUPDIR"/
 exit
fi

if [ "$1" = "individual" ] ; then
	shift
        if [ -z "$1" ] ; then 
          echo "Need group names to perform multiple group analysis."
          exit 1
        fi
	GROUPDIR="$(echo $* | tr ' ' '_' | tr '/' '-')"
	if [ ! -d "$RES_BASE/individual/$GROUPDIR" ]; then
		echo "Individual folder doesn't exist. Creating"
		mkdir -p "$RES_BASE/individual/$GROUPDIR"
                # For individual tracks!
        fi
        if [ ! -d "$RES_BASE/individual/$GROUPDIR/individuals" ]; then
                mkdir -p "$RES_BASE/individual/$GROUPDIR/individuals"
        fi
        if [ ! -d "$DATA_BASE/individual" ]; then
                echo "$DATA_BASE/individual folder doesn't exist. Creating"
                mkdir -p "$DATA_BASE/individual"
        fi
        #if [ ! -e "$RES_BASE/multiple/$GROUPDIR/jquery.min.js" ] ; then
        cp $PROGDIR/resources/jquery.min.js "$RES_BASE/individual/$GROUPDIR"
        #fi
		python3 lin-analysis.py --duration "$DURATION" --time-range-start "$START_TIME" --time-range-end "$END_TIME"  --radius "$RADIUS" "$SAVE_FIGURES" "$SAVE_DATA" --basedir "$BASEDIR" --analysis-mode=individual --individual-analysis all --min-valid-duration "$IND_DURATION" --groups $* --data-folder="$DATA_BASE/individual"/ --figure-folder="$RES_BASE/individual/$GROUPDIR" --movie-folder="$RES_BASE/individual/$GROUPDIR"
 exit
fi

for group in $*; do
	FOLDER=$(echo $group | sed 's/\//-/')
	if [ ! -d "$RES_BASE/$FOLDER" ]; then
		echo "$RES_BASE/$FOLDER folder doesn't exist. Creating"
		mkdir -p "$RES_BASE/$FOLDER"
	fi
	if [ ! -d "$DATA_BASE" ]; then
		echo "$DATA_BASE folder doesn't exist. Creating"
		mkdir -p "$DATA_BASE"
	fi
	python3 lin-analysis.py --duration "$DURATION" --time-range-start "$START_TIME" --time-range-end "$END_TIME"  --radius "$RADIUS" "$SAVE_FIGURES" "$SAVE_DATA" --analysis-mode=group --group-analysis all --group-dir "$BASEDIR"/"$group" --data-folder="$DATA_BASE"/ --figure-folder="$RES_BASE"/"$FOLDER" --movie-folder="$RES_BASE"/"$FOLDER"
done
