#!/bin/bash
convert *.png -coalesce -fuzz 2% -layers optimize +map out.gif