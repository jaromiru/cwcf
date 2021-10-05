#!/usr/bin/env bash
#[1, 0.1, 0.01, 0.001, 0.0001]

tmux new-session -d -s "sepsis-1_0-3" bash ~/cwcf/bin/drl.sh sepsis 1.0 3
tmux new-session -d -s "sepsis-0_1-4" bash ~/cwcf/bin/drl.sh sepsis 0.1 4
tmux new-session -d -s "sepsis-0_01-5" bash ~/cwcf/bin/drl.sh sepsis 0.01 5
tmux new-session -d -s "sepsis-0_001-6" bash ~/cwcf/bin/drl.sh sepsis 0.001 6
tmux new-session -d -s "sepsis-0_0001-7" bash ~/cwcf/bin/drl.sh sepsis 0.0001 7