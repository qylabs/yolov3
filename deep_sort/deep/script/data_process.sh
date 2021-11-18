#!/bin/bash

echo $PWD

data_source_path=data/cropped
target_path=data/cust_person_reid

mkdir -p ${target_path}/gallery
mkdir -p ${target_path}/query

sub_path=${data_source_path}/*
for path in ${sub_path}; do
    echo ${path}
    folder_name=${path##*/} #Remove long prefix
    name=${folder_name::(-2)}
    target_name_path=${target_path}/gallery/${name}
    if [[ ! -e ${target_name_path} ]]; then
        mkdir -p ${target_name_path}
    fi

    if [[ ! -e ${target_path}/query/${name} ]]; then
        mkdir -p ${target_path}/query/${name}
    fi

    cp -r ${path}/*.jpg ${target_name_path}
done