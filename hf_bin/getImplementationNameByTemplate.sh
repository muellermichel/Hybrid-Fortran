#!/bin/bash
architecture=$(echo $1| awk '{print toupper($0)}')
mode=$(echo $2| awk '{print toupper($0)}')
MakesettingsGeneralFile=$3
callGraphFile=$4

source $MakesettingsGeneralFile
defaultImplementation=$(eval "echo \$${architecture}_IMPLEMENTATION_${mode} && : ")
if [ -z "${defaultImplementation+x}" -o "${defaultImplementation}" = "" ] ; then
	defaultImplementation=$(eval "cat ./config/${architecture}_IMPLEMENTATION_${mode} && : ")
fi
if [ -z "${defaultImplementation}" ] ; then
	echo 1>&2 "Please execute './configure' before make" ; exit 1
fi

result=$(printf '{"default":"%s"' "${defaultImplementation}")
templateNames=$(python $HF_DIR/hf/getTemplateNames.py -c $callGraphFile)
templateNamesArr=( $templateNames )
for i in "${!templateNamesArr[@]}"; do
	implementation=$(eval "echo \$${architecture}_IMPLEMENTATION_${mode}_${templateNamesArr[$i]} && : ")
	if [ -n "$implementation" ]; then
		result=${result}$(printf ', "%s":"%s"' "${templateNamesArr[$i]}" "${implementation}")
	fi
done
result=${result}"}"
echo $result