#!/bin/bash

# Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

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

date=`date`
echo "---------------------- testing ${1} for ${2} on ${date} ----------------------" | tee -a ./log.txt 1>&2
configFile="./testConfig_${2}.txt"
if [ ! -e ${configFile} ]; then
	echo "Error: Test configuration file ${configFile} not found."
	exit 1
fi
col_idx=1
argStringsArr=( )
refPostfixesArr=( )
first=true
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

extractionAttempted=false
for i in "${!argStringsArr[@]}"; do
	rm -rf ./out/*.dat
	mkdir -p ./out
	argString=${argStringsArr[$i]}
	refPath=./ref${refPostfixesArr[$i]}/

	if [ "$2" = "valgrind" ]; then
		echo -n "valgrind with parameters${argString},"
		`valgrind --log-file='./log_lastRun.txt' --suppressions=../../../hf_config/valgrind_errors.supp ./${1} ${argString} &>/dev/null`
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

	if [ "$2" = "validation" ]; then
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
				echo "Error with ${2} tests: Reference data directory $refPath not part of the reference data in ./ref.tar.gz" 1>&2
				exit 1
			fi
		fi
	fi
	echo -n "${2} with parameters${argString},"
	timingResult=$(./${1} ${argString} 2>./log_lastRun.txt)
	rc=$?
	if [[ $rc != 0 ]] ; then
		echo "fail"
		echo "Profiled program has returned error code $rc. The output of the last failed run have been logged in 'log_lastRun.txt' in the ${1} test directory."
		echo "--------------------- output of tail log_lastRun.txt -----------------------------"
		tail ./log_lastRun.txt
		echo "----------------------------------------------------------------------------------"
		cat ./log_lastRun.txt >> ./log.txt
	    exit $rc
	fi
	if [ "$2" = "validation" ]; then
		allAccuracy.sh $refPath 2>>./log_lastRun.txt
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
			echo "The output of the last failed run have been logged in 'log_lastRun.txt' in the ${1} test directory."
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
