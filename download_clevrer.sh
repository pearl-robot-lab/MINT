#!/bin/bash
# create a folder to store the downloaded files
if [ ! -d CLEVRER ]; then mkdir CLEVRER; fi
# go to the folder
cd CLEVRER
# define and array with the download links
links=(
"http://data.csail.mit.edu/clevrer/videos/train/video_train.zip" 
"http://data.csail.mit.edu/clevrer/derender_proposals.zip"
)
# download the files
for i in "${links[@]}"
do
    # download the file if it is not already in the folder and there is no folder with the same name
    file_name=$(basename "$i")
    folder_name="${file_name%.*}"
    if [ ! -f "$file_name" ] && [ ! -d "$folder_name" ]; then
        wget "$i"
    fi
    if [ -f "$file_name" ] && [ ! -d "$folder_name" ]; then
        # extract the file
        unzip "$file_name" -d "$folder_name"
        # remove the zip file
        rm "$filename"
    fi
    if [ -d "$folder_name" ]; then
        cd "$folder_name"
        for d in */ ; 
        do
            for f in "$d*.mp4"
            do
                cp $f .
            done
            rm -r $d
        done
        cd ..
    fi
done
# go back to the previous folder
cd ..