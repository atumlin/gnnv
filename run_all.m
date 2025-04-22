%% Script to run verification of GINEConv PF models.

%% Start with getting reachability outputs
% The function is reach_model(modelPath,bus_system,max_snapshots,epsilon)
% Set parameters here.
bus_systems = ["ieee24"]; 
epsilons = [0.005];
max_graphs = 10000; % should be the same number of graphs you trained models on

% Loop over bus systems
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % Get all matching files for this bus system
    model_files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));

    % Loop over epsilon values
    for e = 1:length(epsilons)
        epsilon = epsilons(e);

        % Loop over all matching files
        for f = 1:length(model_files)
            model_path = fullfile("models", model_files(f).name);

            fprintf('Running: %s, epsilon = %.4f\n', model_files(f).name, epsilon);
            reach_model(model_path, bus_system, max_graphs, epsilon);
        end
    end
end

%% Verify reach sets 
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % Get model filenames for this bus system
    files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));

    models = strings(length(files), 1);

    for f = 1:length(files)
        [~, name, ~] = fileparts(files(f).name);
        models(f) = name;
    end

    % Call your verification function
    verify_gine(epsilons, models);
end
