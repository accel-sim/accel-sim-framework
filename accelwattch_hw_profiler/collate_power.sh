#!/bin/bash

SCRIPT_DIR=`pwd`

output_folder="collated_power"

if [ ! "${1}" == "validation_power_reports" ] && [ ! "${1}" == "ubench_power_reports" ]; then
    echo "Please enter a correct power reports directory, one of: [ubench_power_reports,validation_power_reports]. For example: ./collate_power.sh validation_power_reports"
	exit 1
fi

if [ -e "$SCRIPT_DIR/${1}" ] && [ -d "$SCRIPT_DIR/${1}" ]; then
	if [ -d "$output_folder" ]; then
		rm -r "$output_folder"
	fi
	mkdir $output_folder
	for bm in `ls $SCRIPT_DIR/${1}`
	do	
		for data in `ls $SCRIPT_DIR/${1}/$bm`
		do
			power=`cat $SCRIPT_DIR/${1}/$bm/$data | awk -F'Power draw = ' '{print $2}' | awk -F' W' '{print $1}'`
			echo $power >> $output_folder/$bm.rpt
		done
	done
	python gen_hw_power_csv.py $output_folder
    if [  "${1}" == "validation_power_reports" ]; then
        mv hw_power_results.csv hw_power_validation.csv
    else
        mv hw_power_results.csv hw_power_ubench.csv
    fi
else
	echo "Please enter a correct power reports directory. Example: ./collate_power.sh validation_power_reports"
	exit 1
fi


