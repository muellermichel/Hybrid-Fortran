#!/bin/bash

test_dir=~/hybrid/examples/particle/test

RESULTSFILE=$test_dir/results.csv
LOGFILE=$test_dir/log.txt

cd $test_dir
echo "1 Thread C" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	OMP_NUM_THREADS=1 KMP_AFFINITY=compact,verbose $test_dir/particle_c_version_intel >> $RESULTSFILE 2>> $LOGFILE
done

echo "12 Thread C" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	OMP_NUM_THREADS=12 KMP_AFFINITY=compact,verbose $test_dir/particle_c_version_intel >> $RESULTSFILE 2>> $LOGFILE
done

echo "1 Thread HF" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	OMP_NUM_THREADS=1 KMP_AFFINITY=compact,verbose $test_dir/particle_hf_version_cpu >> $RESULTSFILE 2>> $LOGFILE
done

echo "12 Thread HF" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	OMP_NUM_THREADS=12 KMP_AFFINITY=compact,verbose $test_dir/particle_hf_version_cpu >> $RESULTSFILE 2>> $LOGFILE
done

echo "CUDA C" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	$test_dir/particle_cuda >> $RESULTSFILE 2>> $LOGFILE
done

echo "OpenACC C" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	$test_dir/particle_openACC >> $RESULTSFILE 2>> $LOGFILE
done

echo "OpenACC Fortran" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	$test_dir/particle_openACC_Fortran_gpu >> $RESULTSFILE 2>> $LOGFILE
done

echo "HF CUDA Fortran" | tee -a $RESULTSFILE $LOGFILE > /dev/null
for i in {1..3}; do
	$test_dir/particle_hf_version_gpu >> $RESULTSFILE 2>> $LOGFILE
done