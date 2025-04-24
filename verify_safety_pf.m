%% Verify the Safety Critical Specification of PF Models (GINEConv)
% We are developing the specification for looking at the appropriate
% boundaries of voltage magnitude and voltage angle. 
% Author: Anne Tumlin
% Date: 04/24/2025

function verify_safety_pf(epsilons, models)
    for m = 1:length(models)
        modelPath = models(m);

        fprintf('\n--- Starting power safety verification for model: %s ---\n', modelPath);

        % Load model normalization values
        model_data = load("models/" + modelPath + ".mat");
        global_mean_labels = model_data.global_mean_labels;
        global_std_labels = model_data.global_std_labels;

        % Index of voltage magnitude in label (assume 3rd feature)
        voltage_idx = 3;

        % Normalize physical safety bounds (0.9 and 1.1 pu)
        v_min = (0.9 - global_mean_labels(voltage_idx)) / global_std_labels(voltage_idx);
        v_max = (1.1 - global_mean_labels(voltage_idx)) / global_std_labels(voltage_idx);

        for k = 1:length(epsilons)
            eps = epsilons(k);
            fprintf('Processing epsilon: %.4f ...\n', eps);

            % Load reachability results
            rdata = load("results/verified_nodes_" + modelPath + "_eps" + string(eps) + ".mat");

            % Per-node verification
            results = cell(length(rdata.outputSets), 1);
            tic;

            % for i = 1:numel(results)
            for i = 1:50
                Y = rdata.outputSets{i};
                results{i} = verifyVoltageMagnitude(Y, voltage_idx, v_min, v_max);
            end

            elapsed = toc;
            fprintf('Finished epsilon %.4f for model %s in %.2f seconds\n', eps, modelPath, elapsed);

            % Save results in a dedicated subfolder
            safety_results_folder = "results/safety";
            if ~exist(safety_results_folder, 'dir')
                mkdir(safety_results_folder);
            end

            % Save results
            parsave(modelPath, eps, results, rdata.outputSets, rdata.rT, rdata.targets, elapsed)
        end
    end
end

function results = verifyVoltageMagnitude(X, feat_idx, v_min, v_max)
    node = size(X.V, 1); % number of nodes
    results = 3 * ones(node, 1);

    for i = 1:node
        matIdx = zeros(1,node);
        matIdx(i) = 1;

        Y = X.affineMap(matIdx, []);  % Reduce to 1D Star for that feature
        Y = Y.toStar;

        % Create halfspace spec: v_i in [v_min, v_max]
        n_features = size(Y.V, 1);  % Suppose n_features = 4
        selector = zeros(1, n_features);  % [0 0 0 0]
        selector(feat_idx) = 1; % [0 0 1 0] → this isolates feature 3
        
        G = [selector; -selector]; % 2x4 matrix
        % G(1,:) = [0 0 1 0] enforces v3 <= vmax
        % G(2,:) = [0 0 -1 0] enforces -v3 <= -vmin → same as v3 >= vmin
        
        g = [v_max; -v_min];             % [1.1; -0.9]
        
        % Construct halfspaces
        Hs = [HalfSpace(G(1,:), g(1)); HalfSpace(G(2,:), g(2))];

        res = verify_specification(Y, Hs);
        if res == 2
            [lb, ub] = Y.getRanges;
            if any(lb(feat_idx) < v_min) || any(ub(feat_idx) > v_max)
                res = 0;
            else
                res = 1;
            end
        end
        results(i) = res;
    end
end

function parsave(modelPath, epsilon, results, outputSets, rT, targets, timing)
    fname = "results/safety/safety_verified_nodes_" + modelPath + "_eps" + string(epsilon) + ".mat";
    save(fname, "results", "outputSets", "rT", "targets", "timing");
end