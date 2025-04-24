%% Run All Verification Results

% Set parameters here.
bus_systems = ["ieee24","ieee39"];
epsilons = [0.001, 0.1];
features = [1,2];

% Perturb all features. Get reachable sets. Verify both properties. 
run_verification(bus_systems,epsilons);

% Perturb specific features. Get reachable sets. Verify both properties. 
run_verification_sp(bus_systems,epsilons,features);