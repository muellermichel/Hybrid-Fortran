#!/bin/bash

# Copyright (C) 2015 Michel Müller, Tokyo Institute of Technology

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

set -e

input_file=${1}
output_file=${2}
metadata_file=${3}
build_dir=${4}
build_dir_sources=${5}
preprocessor_args=${6}
option_flags=${7}

conversion_command="python ${HF_DIR}/hf_processor/generateF90fromH90AndAnalyzedCallGraph.py -i ${input_file} -c ${metadata_file} --implementation=${build_dir}implementationNamesByTemplate --optionFlags=${option_flags} ${preprocessor_args} > ${build_dir_sources}hf_temp.P90"
echo "$conversion_command"
eval "$conversion_command"
if [ ! -e ${output_file} ]; then
	echo "creating new file ${output_file}"
	mv ${build_dir_sources}hf_temp.P90 ${output_file}
else
 	diff "${build_dir_sources}hf_temp.P90" "${output_file}" > /dev/null 2>&1 && :
	diff_result=$?
	if [ $diff_result -eq 1 ]; then
		echo "updating file ${output_file}"
		mv ${build_dir_sources}hf_temp.P90 ${output_file}
	else
		echo "no change in file ${output_file}"
		rm ${build_dir_sources}hf_temp.P90
	fi
fi