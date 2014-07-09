#!/bin/bash
function run {
	wrf_executable_name=$1
	echo "running $wrf_executable_name" 1>&2
	rm -f $wrf_executable_name.log
	set +e
	./$wrf_executable_name.exe > $wrf_executable_name.log 2>&1
	rc=$?
	if [ $rc -ne 0 ] ; then
		echo "$wrf_executable_name.exe has returned error $rc. Print last 20 lines of $wrf_executable_name.log:" 1>&2
		tail -n 20 $wrf_executable_name.log
		exit $rc
	fi
	set -e
}

set -e

prev_dir=$(pwd)
cd "$(dirname "$0")"

architecture=$1
# rm -f ./out/x.dat ./out/y.dat
# ./particle_hf_version_$architecture
rm -f ./out/x.dat ./out/y.dat
./particle_openACC_Fortran_$architecture

cd $prev_dir