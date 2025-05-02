%% Script to run verification of GINEConv PF models w/ specific perturbations
% Here we are applying specific feature perturbations within the
% GINEConv PF models. Then, we are verifying whether the perturbations will
% push to make predictions outside of a specified boundary (based on RMSE).
% Finally, we also check whether these perturbations will cause voltage
% magnitude to go outside of operational limits. 

function run_verification_sp(bus_systems,epsilons,features)

fprintf("\n----------- GNN Verification on Specific Features ---------------\n");
specific_perturbation = true;

% Loop over bus systems
for b = 1:length(bus_systems)
    bus_system = bus_systems(b);

    % Get all matching files for this bus system
    model_files = dir(fullfile("models", "gcn_" + bus_system + "_*.mat"));
    models = strings(length(model_files), 1);
    
    % Create reachable sets for specific perturbed features
    for f = 1:length(model_files)
        model_path = fullfile("models", model_files(f).name);
        [~, name, ~] = fileparts(model_files(f).name);
        models(f) = name;
        
        fprintf("\n--- Creating reachable sets for model: %s ---\n", model_files(f).name);

        % Loop over epsilon values
        for e = 1:length(epsilons)
            epsilon = epsilons(e);
            fprintf('Running epsilon: %.4f\n', epsilon);
            reach_pf_gine_specific_perturb(model_path,epsilon,features);
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