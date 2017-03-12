#!/usr/bin/env bash
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0

python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 5 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 5 0 5 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 0 5 5 0 0 0 0 0 0 0 0 0 0 0 0
python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0

