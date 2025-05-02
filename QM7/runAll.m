% 1) Train
% train_multiple_reluGNN;

seeds = [0,1,2,3,4]; % models
% seeds = [5,6,7,8,9]; % models
epsilon = [0.005, 0.01, 0.02, 0.05]; % attack


% 2) Reachability
reach_all_models_Linf(seeds,epsilon);

% 3) Verify
verify_AllReachSets(seeds,epsilon);

% 4) Process results
process_results(seeds,epsilon);