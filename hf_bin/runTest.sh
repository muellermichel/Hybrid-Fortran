#!/bin/bash

# Copyright (C) 2013 Michel Müller (Typhoon Computing), RIKEN Advanced Institute for Computational Science (AICS)

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

if [ -z "$output_file_pattern" ]; then
	output_file_pattern="./out/*.dat"
fi
date=`date`
echo "------- testing ${executable_name} for ${configuration_name} on ${architecture} ; ${date} in `pwd` ; output pattern: ${output_file_pattern} -------" | tee -a ./log.txt 1>&2
configFile="./testConfig_${configuration_name}.txt"
argStringsArr=( )
refPostfixesArr=( )
if [ -e ${configFile} ]; then
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
	argStringsArr=( "" )
	refPostfixesArr=( "" )
fi

extractionAttempted=false
for i in "${!argStringsArr[@]}"; do
	rm -rf $output_file_pattern
	mkdir -p "$(dirname $output_file_pattern)"
	argString=""
	if [[ $executable_name == "*${architecture}*" ]] ; then
		argString=${argStringsArr[$i]}
	else
		argString="${architecture} ${argStringsArr[$i]}"
	fi
	refPath=./ref${refPostfixesArr[$i]}/

	if [ "$configuration_name" = "valgrind" ]; then
		echo -n "valgrind with parameters${argString},"
		`valgrind --log-file='./log_lastRun.txt' --suppressions=../../../hf_config/valgrind_errors.supp ./${executable_name} ${argString} &>/dev/null`
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
			tail ./log_lastRun.txt
			echo "----------------------------------------------------------------------------------"
			cat ./log_lastRun.txt >> ./log.txt
			exit 1
		fi
		continue
	fi

	if [ "$configuration_name" = "validation" ]; then
		if [ ! -e $refPath ]; then
			if ! $extractionAttempted; then
				echo "extracting reference data"
		    	tar -xzvf ./ref.tar.gz > /dev/null
		    	rc=$?
		    	if [[ $rc != 0 ]] ; then
		    		exit 1
		    	fi
		    	extractionAttempted=true
			fi
			if [ ! -e $refPath ]; then
				echo "Error with ${configuration_name} tests: Reference data directory $refPath not part of the reference data in ./ref.tar.gz" 1>&2
				exit 1
			fi
		fi
	fi
	echo -n "$calling ${executable_name} ( with parameters ${argString} ) for ${configuration_name} ,"
	timingResult=$(./${executable_name} ${argString} 2>./log_lastRun.txt)
	rc=$?
	if [[ $rc != 0 ]] ; then
		echo "fail"
		echo "Profiled program has returned error code $rc. The error output of the last failed run have been logged in 'log_lastRun.txt' in the ${executable_name} test directory."
		echo "stdout: $timingResult"
		echo "--------------------- output of tail log_lastRun.txt -----------------------------"
		tail ./log_lastRun.txt
		echo "----------------------------------------------------------------------------------"
		cat ./log_lastRun.txt >> ./log.txt
	    exit $rc
	fi
	if [ "$configuration_name" = "validation" ]; then
		${HF_DIR}/hf_bin/allAccuracy.sh $refPath "$output_file_pattern" $source_before $source_after 2>>./log_lastRun.txt
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
			tail ./log_lastRun.txt
			echo "----------------------------------------------------------------------------------"
			exit 1
		fi
	else
		echo "${timingResult}"
		cat ./log_lastRun.txt >> ./log.txt
	fi
done
exit 0
