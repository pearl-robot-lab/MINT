#!/bin/bash
convert *.png -coalesce -dispose Background -fuzz 2% -layers optimize +map out.gif
