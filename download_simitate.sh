#! /bin/bash
# create a folder to store the downloaded files
if [ ! -d SIMITATE ]; then mkdir SIMITATE; fi
# go to the folder
cd SIMITATE
# define and array with the download links
declare -a links=(
" "
"https://agas.uni-koblenz.de/data/simitate/data/simitate/data/extracted/sequential/bring/"
"https://agas.uni-koblenz.de/data/simitate/data/simitate/data/extracted/sequential/rearrange/"
"https://agas.uni-koblenz.de/data/simitate/data/simitate/data/extracted/sequential/pick_and_place/"
"https://agas.uni-koblenz.de/data/simitate/data/simitate/data/extracted/motions/stack/"
)
# define a counter
counter=0
images_folder="_kinect2_qhd_image_color_rect_compressed/"
# download the files requested by the user in arguments
for i in "$@"
do
    counter=0
    # get the link from the array
    link=${links[$i]}
    # get the subdirectories from the link
    subdirs=$(curl -s $link | grep -o '<a href="[^"]*">' | sed 's/<a href="//g' | sed 's/">//g' )
    # iterate over the subdirectories
    for subdir in $subdirs
    do
        # skip if subdir has ? in its name or starts with data
        if [[ $subdir == *"?"* ]] || [[ $subdir == *"data"* ]]; then continue; fi
        # get the files from the subdirectory
        files=$(curl -s $link$subdir | grep -o '<a href="[^"]*">' | sed 's/<a href="//g' | sed 's/">//g')
        # iterate over the files
        for file in $files
        do
            # skip if file has ? in its name or starts with data
            if [[ $file == *"?"* ]] || [[ $file == *"data"* ]]; then continue; fi
            # increase the counter by 1
            ((counter++))
            # the name of the file after adding the subfolder name
            link_final=$link$subdir$file$images_folder
            # create a direcotry to store the images if it does not exist
            if [ ! -d $(basename $link)_$counter ]; then mkdir $(basename $link)_$counter; fi
            cd $(basename $link)_$counter
            # download the conent of link final
            wget -r -np -nH --cut-dirs=11 --reject index.html* $link_final
            cd ..
        done
    done
done
# go back to the previous folder
cd ..