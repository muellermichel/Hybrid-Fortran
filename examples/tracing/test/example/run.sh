#!/bin/bash
set -e

ARCH=$1
MODE=$2

./example_$ARCH > log_temp.txt 2>&1 && :
rc=$?
cat log_temp.txt 1>&2
if [[ "$ARCH" == "gpu" ]] ; then
	echo "GPU run has returned error code $rc" 1>&2
	count=$(cat log_temp.txt | grep -i -e 'does not match the data' 2>&1 | wc -l)
	if [ $count -ne 1 ] ; then
    	echo "==> GPU Trace has more failures than expected" 1>&2
    	exit 102
    else
    	exit 0
    fi
	cat log_temp.txt | grep -i -e 'In module example, subroutine faulty_wrapper_real_4' && :
    errorgrep_rc=$?
    if [[ $errorgrep_rc != 0 ]] ; then
    	echo "==> GPU run has not failed at the expected place" 1>&2
    	exit 101
    else
    	exit 0
    fi
fi
exit $rc
