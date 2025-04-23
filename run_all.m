%% Script to run verification of GINEConv PF models.

%% Start with getting reachability outputs
% The function is reach_model(modelPath,bus_system,max_snapshots,epsilon)
% Set parameters here.
bus_systems = ["ieee24"]; 
epsilons = [0.005];

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
            reach_model(model_path, epsilon);
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

%% Process Results 
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % Get model filenames for this bus system
    files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));
    models = strings(length(files), 1);

    for f = 1:length(files)
        [~, model_name, ~] = fileparts(files(f).name);
        models(f) = model_name;
    end

    for m = 1:length(models)
        model_path = models(m);
        eN = length(epsilons);

        robust_counts    = zeros(eN,1);
        unknown_counts   = zeros(eN,1);
        notrobust_counts = zeros(eN,1);
        total_outputs    = zeros(eN,1);
        timings          = NaN(eN,1); 

        for k = 1:eN
            eps = epsilons(k);
            result_file = "results/verified_nodes_" + model_path + "_eps" + string(eps) + ".mat";

            if exist(result_file, 'file') == 2
                data = load(result_file);  % loads: results, targets, rT, timing
                results = data.results;

                for i = 1:length(results)
                    res = results{i};
                    robust_counts(k)    = robust_counts(k)    + sum(res == 1);
                    unknown_counts(k)   = unknown_counts(k)   + sum(res == 2);
                    notrobust_counts(k) = notrobust_counts(k) + sum(res == 0);
                    total_outputs(k)    = total_outputs(k)    + numel(res);
                end

                if isfield(data, 'timing')
                    timings(k) = data.timing;
                end
            else
                warning("Missing result file: %s", result_file);
            end
        end

        % Save .mat summary with timing
        save("results/summary_Linf_" + model_path + ".mat", ...
            "robust_counts", "unknown_counts", "notrobust_counts", ...
            "total_outputs", "timings");

        % Write human-readable text summary
        model_data = load("models/" + model_path + ".mat");
        summary_txt = "results/summary_Linf_" + model_path + ".txt";
        fileID = fopen(summary_txt, 'w');
        fprintf(fileID, 'Robustness Summary for Model: %s\n', model_path);
        % fprintf(fileID, 'Model accuracy: %.4f\n\n', model_data.accuracy);
        fprintf(fileID, 'Epsilon | Robust   Unknown   NotRobust   Total    Time (sec)\n');
        for k = 1:eN
            if total_outputs(k) > 0
                fprintf(fileID, ' %.4f  |  %.3f    %.3f     %.3f     %d     %.2f\n', ...
                    epsilons(k), ...
                    robust_counts(k)/total_outputs(k), ...
                    unknown_counts(k)/total_outputs(k), ...
                    notrobust_counts(k)/total_outputs(k), ...
                    total_outputs(k), ...
                    timings(k));
            else
                fprintf(fileID, ' %.4f  |   N/A      N/A       N/A       0     N/A\n', epsilons(k));
            end
        end
        fclose(fileID);
    end
end