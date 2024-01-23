#!/bin/bash

# specify the directory
dir="/lustre/work/wachslab/aredwine3/VAME_working/real_videos/“  

# at the directory 
cd $dir

# loop through each file in the directory
for file in *; do
    # get the base name and the extension
    base_name="${file%.*}"
    extension="${file##*.}"

    # rename the file
    mv "$file" "${base_name}_0min_to_15min.$extension"
done