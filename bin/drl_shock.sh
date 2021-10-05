#!/usr/bin/env bash
#[1, 0.1, 0.01, 0.001, 0.0001]

tmux new-session -d -s "shock-1_0-3" bash ~/cwcf/bin/drl.sh shock 1.0 3
tmux new-session -d -s "shock-0_1-4" bash ~/cwcf/bin/drl.sh shock 0.1 4
tmux new-session -d -s "shock-0_01-5" bash ~/cwcf/bin/drl.sh shock 0.01 5
tmux new-session -d -s "shock-0_001-6" bash ~/cwcf/bin/drl.sh shock 0.001 6
tmux new-session -d -s "shock-0_0001-7" bash ~/cwcf/bin/drl.sh shock 0.0001 7