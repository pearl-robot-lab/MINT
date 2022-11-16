#!/bin/bash
for d in */ ; do
    cd "$d"
    montage -density 500 -tile 5x2 -geometry -10-30 $(ls *.png | sort -V) out.png
    cd ..
done