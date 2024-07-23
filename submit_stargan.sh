#!/bin/bash

if [[ $# -ne 2 ]] ; then
	echo "Incorrect number of arguments."
	exit 1
fi

echo "Loading branch: $1"
rm -rf "./stargan-experiments/${1}"
git clone -b $1 https://github.com/googlyeyesultra/stargan-experiments.git "./stargan-experiments/${1}" 

condor_submit stargan.sub -a "arguments=$1 $2" -a "log=./condor_out/log.${1}_${2}" -a "error=./condor_out/err.${1}_${2}" -a "output=./condor_out/out.${1}_${2}"