#!/bin/bash

for d in *.mp4 ; do
    echo $d
    ffmpeg -y -i $d -filter_complex "fps=5,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer" "$d.gif"
done