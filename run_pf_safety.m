%% Script to run safety specification verification of GINEConv PF models.

%% Set parameters here.
bus_systems = ["ieee24"];
epsilons = [0.001, 0.1];
% epsilons = [0.001, 0.01, 0.05, 0.1];

% Loop over bus systems
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % % Get all matching files for this bus system
    % model_files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));
    % 
    % % Loop over epsilon values
    % for e = 1:length(epsilons)
    %     epsilon = epsilons(e);
    % 
    %     % Loop over all matching files
    %     for f = 1:length(model_files)
    %         model_path = fullfile("models", model_files(f).name);
    % 
    %         fprintf('Running: %s, epsilon = %.4f\n', model_files(f).name, epsilon);
    %         reach_model(model_path, epsilon);
    %     end
    % end

    % % Get model filenames for this bus system
    % files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));
    % models = strings(length(files), 1);
    % for f = 1:length(files)
    %     [~, name, ~] = fileparts(files(f).name);
    %     models(f) = name;
    % end
    % 
    % % Call safety specification verification function
    % verify_safety_pf(epsilons, models);

    % Process safety results for each model
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
            result_file = "results/safety/safety_verified_nodes_" + model_path + "_eps" + string(eps) + ".mat";

            if exist(result_file, 'file') == 2
                data = load(result_file);  % loads: results, outputSets, targets, elapsed, eps
                results = data.results;

                for i = 1:length(results)
                    res = results{i};
                    robust_counts(k)    = robust_counts(k)    + sum(res == 1);
                    unknown_counts(k)   = unknown_counts(k)   + sum(res == 2);
                    notrobust_counts(k) = notrobust_counts(k) + sum(res == 0);
                    total_outputs(k)    = total_outputs(k)    + numel(res);
                end

                timings(k) = data.timing;
            else
                warning("Missing result file: %s", result_file);
            end
        end

        % Save .mat summary
        save("results/safety/summary_safety_" + model_path + ".mat", ...
            "robust_counts", "unknown_counts", "notrobust_counts", ...
            "total_outputs", "timings");

        % Write human-readable text summary
        summary_txt = "results/safety/summary_safety_" + model_path + ".txt";
        fileID = fopen(summary_txt, 'w');
        fprintf(fileID, 'Voltage Magnitude Safety Summary for Model: %s\n\n', model_path);
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
