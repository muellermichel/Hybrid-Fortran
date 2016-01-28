#!/bin/bash
set -e

# Copyright (C) 2016 Michel Müller, Tokyo Institute of Technology

# This file is part of Hybrid Fortran.

# Hybrid Fortran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Hybrid Fortran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.

executable_name=${1}
architecture=${2}
configuration_name=${3}
output_file_pattern=${4}
source_before=${5}
source_after=${6}
formatParam="${7}"

working_dir=$(pwd)

if [ -z "$output_file_pattern" ]; then
	output_file_pattern="./out/*.dat"
fi
if [ -z "$configuration_name" ]; then
	configuration_name="normal"
fi
date=`date`
echo "------- testing ${executable_name} for ${configuration_name} on ${architecture} ; ${date} in ${working_dir} ; output pattern: ${output_file_pattern} -------" | tee -a ./log.txt 1>&2
configFile="./testConfig_${configuration_name}.txt"
configFileParams="./testConfigParams_${configuration_name}.txt"
argStringsArr=( )
refPostfixesArr=( )
if [ -e ${configFileParams} ]; then
	echo "paramater config file ${configFileParams} found" 1>&2
	while read line || [[ -n $line ]]; do
		argStringsArr+=("$line")
		refPostfixesArr+=(_`echo $line | tr " " "_"`)
	done <${configFileParams}
elif [ -e ${configFile} ]; then
	echo "config file ${configFile} found" 1>&2
	col_idx=1
	while true
	do
		argNames=`cut -d' ' -f${col_idx} ${configFile}`
		val_idx=$(( $col_idx + 1 ))
		if [[ ! $argNames ]]; then
			break
		fi
		argVals=`cut -d' ' -f${val_idx} ${configFile}`
		if [[ ! $argVals ]]; then
			echo "Error reading ${configFile}: arguments without value." 1>&2
			exit 1
		fi
		argNamesArr=($argNames)
		argValsArr=($argVals)
		if [ ${#argNamesArr[@]} -ne ${#argValsArr[@]} ]; then
			echo "Error reading ${configFile}: all arguments need to have an assigned value." 1>&2
			exit 1
		fi
		for i in "${!argNamesArr[@]}"; do
			if [ ${#argStringsArr[@]} -lt ${i} ]; then
				argStringsArr+=("-${argNamesArr[$i]} ${argValsArr[$i]}")
				refPostfixesArr+=("_${argNamesArr[$i]}${argValsArr[$i]}")
			else
				argStringsArr[$i]="${argStringsArr[$i]} -${argNamesArr[$i]} ${argValsArr[$i]}"
				refPostfixesArr[$i]="${refPostfixesArr[$i]}_${argNamesArr[$i]}${argValsArr[$i]}"
			fi
		done
		col_idx=$(( $col_idx + 2 ))
	done
else
	echo "no config file found ($configFileParams or $configFile) - starting executable without command line parameters" 1>&2
	argStringsArr=( "" )
	refPostfixesArr=( "" )
fi

rm -rf $output_file_pattern
output_file_pattern_list=( ${output_file_pattern} )
mkdir -p "$(dirname output_file_pattern_list[0])"

extractionAttempted=false
referenceAvailable=false
for i in "${!argStringsArr[@]}"; do
	argString=""
	if [[ $executable_name == "*${architecture}*" ]] ; then
		argString=${argStringsArr[$i]}
	else
		argString="${architecture} ${argStringsArr[$i]}"
	fi
	refPath=./ref${refPostfixesArr[$i]}/

	if [[ "$configuration_name" = "valgrind" ]] && [[ -n `file ./${executable_name} | grep -i "executable"` ]]; then
		echo -n "valgrind with parameters${argString},"
		touch ./log_lastRun.txt
		echo "running valgrind with: " >> ./log_lastRun.txt
		echo "valgrind --log-file='./log_lastRun.txt' --suppressions=$HF_DIR/hf_config/valgrind_errors.supp ./${executable_name} ${argString} &>/dev/null" >> ./log_lastRun.txt
		valgrind --log-file='./log_lastRun.txt' --suppressions=$HF_DIR/hf_config/valgrind_errors.supp ./${executable_name} ${argString} &>/dev/null && :
		cat ./log_lastRun.txt 2>&1 | grep 'Unrecognised instruction' &> './log_temp.txt'
		if [[ -s './log_temp.txt' ]]; then
			echo "fail"
			echo "Error trying to execute valgrind: Program code not compatible. Please make sure to only apply the valgrind test to CPU code that has been compiled with debugging parameters (e.g. 'make build_cpu DEBUG=1')"
			cat ./log_lastRun.txt >> ./log.txt
			exit 1
		fi
		valgrindResult=$(cat ./log_lastRun.txt | grep 'ERROR SUMMARY' | cut -d' ' -f4)
		if [[ $valgrindResult -eq 0 ]]; then
			echo "pass"
		else
			echo "fail"
			echo "The output of the last failed run have been logged in 'log_lastRun.txt'"
			echo "--------------------- output of tail log_lastRun.txt -----------------------------"
			tail -n 30 ./log_lastRun.txt
			echo "----------------------------------------------------------------------------------"
			cat ./log_lastRun.txt >> ./log.txt
			exit 1
		fi
		continue
	fi

	if [ "$configuration_name" = "validation" ]; then
		if [ -e ./ref.tar.gz ] && [ ! -e $refPath ]; then
			if ! $extractionAttempted; then
				echo "extracting reference data"
				tar -xzvf ./ref.tar.gz > /dev/null
				extractionAttempted=true
			fi
			if [ -e ./ref.tar.gz ] && [ ! -e $refPath ]; then
				echo "Error with ${configuration_name} tests: Reference data directory $refPath not part of the reference data in ./ref.tar.gz" 1>&2
				exit 1
			fi
		fi
	fi

	remoteCallPrefix=""

	echo -n "calling ${executable_name} ( with parameters ${argString} ) for ${configuration_name} ,"
	if [ -z "$HF_RUN_OVER_SSH" ]; then
		timingResult=$(./${executable_name} ${argString} 2>./log_lastRun.txt && :)
		rc=$?
	else
		timingResult=$(ssh $HF_RUN_OVER_SSH "cd ${working_dir} && ./${executable_name}" ${argString} 2>./log_lastRun.txt && :)
		rc=$?
	fi
	if [[ $rc != 0 ]] ; then
		echo "fail"
		echo "Profiled program has returned error code $rc. The error output of the last failed run have been logged in 'log_lastRun.txt' in the ${executable_name} test directory."
		echo "stdout: $timingResult"
		echo "--------------------- output of tail log_lastRun.txt -----------------------------"
		tail -n 30 ./log_lastRun.txt
		echo "----------------------------------------------------------------------------------"
		cat ./log_lastRun.txt >> ./log.txt
	    exit $rc
	fi
	cat ./log_lastRun.txt | grep -i -e 'fatal error' && :
	errorgrep_rc=$?
	if [[ $errorgrep_rc != 1 ]] ; then
		echo "fail"
		echo "Profiled program has logged a fatal error. The error output of the last failed run have been logged in 'log_lastRun.txt' in the ${executable_name} test directory."
		echo "stdout: $timingResult"
		echo "--------------------- output of tail log_lastRun.txt -----------------------------"
		tail -n 30 ./log_lastRun.txt
		echo "----------------------------------------------------------------------------------"
		cat ./log_lastRun.txt >> ./log.txt
	    exit 102
	fi
	if [ "$configuration_name" = "validation" ] && [ -e $refPath ]; then
		${HF_DIR}/hf_bin/allAccuracy.sh "$refPath" "$output_file_pattern" "$source_before" "$source_after" "$formatParam" 2>>./log_lastRun.txt && :
		rc=$?
		validationResult=""
		cat ./log_lastRun.txt >> ./log.txt
		if [[ $rc != 0 ]] ; then
		    validationResult="fail"
		else
			validationResult="pass"
		fi
		echo "${timingResult}",$validationResult
		if [[ $rc != 0 ]] ; then
			echo "fail"
			echo "The output of the last failed validation has been logged in 'log_lastRun.txt' in the ${executable_name} test directory."
			echo "--------------------- output of tail log_lastRun.txt -----------------------------"
			tail -n 30 ./log_lastRun.txt
			echo "----------------------------------------------------------------------------------"
			exit 1
		fi
	else
		echo "${timingResult}"
		cat ./log_lastRun.txt >> ./log.txt
	fi
done
exit 0
