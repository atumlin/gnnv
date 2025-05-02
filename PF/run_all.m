%% Run All Verification Results

% Set parameters here.
bus_systems = ["ieee24"];
epsilons = [0.01,0.1,0.25,0.5,0.75];
features = [1,2];

% Perturb all features. Get reachable sets. Verify both properties. 
run_verification(bus_systems,epsilons);

% Perturb specific features. Get reachable sets. Verify both properties. 
% run_verification_sp(bus_systems,epsilons,features);