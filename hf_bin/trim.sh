#!/bin/bash
set -e

input=$1
no_external_space="$(echo -e "${input}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
echo -e $no_external_space