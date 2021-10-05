#!/usr/bin/env bash
#[1, 0.1, 0.01, 0.001, 0.0001]

tmux new-session -d -s "sepsis_1.0_3" bash ~/cwcf/bin/drl.sh sepsis 1.0 3
tmux new-session -d -s "sepsis_0.1_4" bash ~/cwcf/bin/drl.sh sepsis 0.1 4
tmux new-session -d -s "sepsis_0.01_5" bash ~/cwcf/bin/drl.sh sepsis 0.01 5
tmux new-session -d -s "sepsis_0.001_6" bash ~/cwcf/bin/drl.sh sepsis 0.001 6
tmux new-session -d -s "sepsis_0.0001_7" bash ~/cwcf/bin/drl.sh sepsis 0.0001 7