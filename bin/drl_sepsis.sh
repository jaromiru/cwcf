#!/usr/bin/env bash
#[1, 0.1, 0.01, 0.001, 0.0001]

tmux new-session -d -s "sepsis_1.0_0" bash ~/cwcf/bin/drl.sh sepsis 1.0 0
tmux new-session -d -s "sepsis_0.1_1" bash ~/cwcf/bin/drl.sh sepsis 0.1 1
tmux new-session -d -s "sepsis_0.01_2" bash ~/cwcf/bin/drl.sh sepsis 0.01 2
tmux new-session -d -s "sepsis_0.001_3" bash ~/cwcf/bin/drl.sh sepsis 0.001 3
tmux new-session -d -s "sepsis_0.0001_4" bash ~/cwcf/bin/drl.sh sepsis 0.0001 4