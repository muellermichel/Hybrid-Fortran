#!/bin/bash
./${1}_${2} ${3} 2> /dev/null | grep "${3}\W" | cat -b
