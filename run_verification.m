%% Script to run verification of GINEConv PF models.
% Here we are applying perturbations across all features within the
% GINEConv PF models. Then, we are verifying whether the perturbations will
% push to make predictions outside of a specified boundary (based on RMSE).
% Finally, we also check whether these perturbations will cause voltage
% magnitude to go outside of operational limits. 

function run_verification(bus_systems,epsilons)

fprintf("\n----------- GNN Verification ---------------\n");
specific_perturbation = false;

% Loop over bus systems
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % Get all matching files for this bus system
    model_files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));
    models = strings(length(model_files), 1);
    
    % Create reachability outputs per model and epsilon
    for f = 1:length(model_files)
        model_path = fullfile("models", model_files(f).name);
        [~, name, ~] = fileparts(model_files(f).name);
        models(f) = name;

        fprintf("\n--- Creating reachable sets for model: %s ---\n", model_files(f).name);

        % Loop over epsilon values
        for e = 1:length(epsilons)
            epsilon = epsilons(e);
            fprintf('Running epsilon: %.4f\n', epsilon);
            % Create reach model here
            reach_pf_gine(model_path, epsilon);
        end
    end
    
    % Call the verification function
    verify_pf_gine(epsilons, models, specific_perturbation);

    % Process verification results
    process_pf_results(epsilons, models, specific_perturbation);

    % Call safety specification verification function
    verify_pf_gine_volt_magn(epsilons, models, specific_perturbation);

    % Process safety results for each model
    process_volt_magn_results(epsilons, models, specific_perturbation);
end

end