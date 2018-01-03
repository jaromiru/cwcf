load run_0.dat; load run_1.dat ; load run_2.dat ; load run_perf.dat

figure('units','normalized','position',[.1 .1 .8 .8]);
subplot(3,1,1); plot(run_0); title('State_0');
subplot(3,1,2); plot(run_1); title('State_A');
subplot(3,1,3); plot(run_2); title('State_B');

figure('units','normalized','position',[.1 .1 .8 .8]);
subplot(3,1,1); plot(run_perf(:, 1)); title('Average reward');
subplot(3,1,2); plot(run_perf(:, 2)); title('Average length');
subplot(3,1,3); plot(run_perf(:, 3)); title('Average correct');
