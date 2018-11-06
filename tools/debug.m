load run_val_0.dat; load run_val_perf.dat; load run_trn_perf.dat

% f = figure('units','normalized','position',[0 0 1 1]);
f = figure('units','pixels','position',[0 0 400 800]);
set(f, "defaultlinelinewidth", 1);

subplot(5,1,1); plot(run_val_0); title('Q-values in s_0');

subplot(5,1,2); hold on; 
plot(run_trn_perf(:, 1), 'g');
plot(run_val_perf(:, 1)); 
title('Reward');
legend("training", "validation")
legend("boxoff")


subplot(5,1,3); hold on; 
plot(run_val_perf(:, 2), 'b'); plot(run_val_perf(:, 3), 'r'); 
title('Length / Cost');
legend("length", "cost")
legend("boxoff")

subplot(5,1,4); area(run_val_perf(:, 4)); title('HPC usage');

subplot(5,1,5); hold on; 
plot(run_trn_perf(:, 5), 'g');
plot(run_val_perf(:, 5));
title('Accuracy');
legend("training", "validation")
legend("boxoff")