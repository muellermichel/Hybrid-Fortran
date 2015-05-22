#!/bin/bash
set -e

#print the filename from the given path
get_file_name(){
	echo $(basename $1)
}

#print the filename extension
get_file_name_extension(){
	local filename=$(get_file_name $1)
	extension_candidate=${filename##*.}
	if [ "${extension_candidate}" != "${filename}" ] ; then
		echo ${extension_candidate}
	else
		echo ""
	fi
}

executable_list=( $1 )
install_path_list=( $2 )

if [ "$3" != "" ]; then
	postfix="_$3"
else
	postfix=""
fi

index=0
target_paths=""
for executable in ${executable_list[*]} ; do
	target_path=""
	if [ "${#install_path_list[@]}" -lt 1 ]; then
		target_path=${executable}${postfix}
	else
		if [ "${#install_path_list[@]}" -gt "${index}" ]; then
			install_path=${install_path_list[$index]}
		else
			install_path=${install_path_list[$((${#install_path_list[@]} - 1))]}
		fi
		extension=$(get_file_name_extension ${executable})
		if [ "${extension}" != "" ] ; then
			filename_with_extension=${executable##*/}
			filename=${filename_with_extension%.*}
			target_path=${install_path}/${filename}${postfix}.${extension}
		else
			filename=${executable##*/}
			target_path=${install_path}/${filename}${postfix}
		fi
	fi
	target_paths="${target_paths} ${target_path}"
	index=$[index + 1]
done

echo "${target_paths}"

