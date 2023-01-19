#!/bin/bash
# create a folder to store the downloaded files
if [ ! -d datasets ]; then mkdir datasets; fi
# go to the folder
cd datasets
# define and array with the download links
links=(
"https://zenodo.org/record/7105244/files/CLEVRER_20videos_128frames_480x320_20training_80validation.pt"
"https://zenodo.org/record/7062781/files/Simitate_80videos_100frames_480x272_4tasks_20training_5validation.pt"
"https://zenodo.org/record/7062781/files/MIME_80videos_50frames_320x240_4tasks_20training_5validation.pt"
# "https://zenodo.org/record/6568081/files/MIME_dataset_320x240_2tasks_3training_1validation.pt" 
# "https://zenodo.org/record/6568081/files/Simitate_dataset_480x272_2tasks_3training_1validation.pt"
)
# download the files
for i in "${links[@]}"
do
    # download the file if it is not already in the folder
    if [ ! -f "$i" ]; then
        wget "$i"
    fi
done
# go back to the previous folder
cd ..