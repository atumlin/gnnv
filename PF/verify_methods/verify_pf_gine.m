%% Verify the robustness of GINEConv models 
% This code is specifically looking at PF Analysis using GINEConv Layers.
% It is verifying the GINE layers within some boundary defined by RMSE of
% the models. 
% Author: Anne Tumlin
% Date: 04/02/2025

function verify_pf_gine(epsilons, models, specific_perturbation)
    for m = 1:length(models)
        modelPath = models(m);

        fprintf('\n--- Starting verification for model: %s ---\n', modelPath);

        % Extract timestamp from model filename
        tokens = regexp(modelPath, 'gcn_(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', 'tokens');
        bus_system = tokens{1}{1};
        timestamp = tokens{1}{2};
        rmse = get_rmse_from_log(bus_system, timestamp);
        
        for k = 1:length(epsilons)
            eps = epsilons(k);
            fprintf('Processing epsilon: %.4f ...\n', eps);
            % Load results
            if specific_perturbation
                rdata = load("results/gine_sp/verified_nodes_" + modelPath + "_eps" + string(eps) + ".mat");
            else
                rdata = load("results/gine/verified_nodes_" + modelPath + "_eps" + string(eps) + ".mat");
            end

            % Per-node verification
            results = cell(length(rdata.outputSets), 1);
            tic; 
            delta = rmse*2; 

            % for i = 1:numel(results)
            for i = 1:100
                Y = rdata.outputSets{i};
                label = rdata.targets{i};
                results{i} = verifyNode(Y, label, delta);
            end

            elapsed = toc;
            fprintf('Finished epsilon %.4f for model %s in %.2f seconds\n', eps, modelPath, elapsed);

            % Save results
            parsave(modelPath, eps, results, rdata.outputSets, rdata.rT, rdata.targets, elapsed, rmse, specific_perturbation);
        end
    end
end

% Check robustness of every node. 
function results = verifyNode(X, target,delta)
node = size(target,1);
results = 3*ones(node,1); 
    for i=1:node
        matIdx = zeros(1,node);
        matIdx(i) = 1;
        Y = X.affineMap(matIdx, []); % Reduce Imagestar to 1 dimension
        Y = Y.toStar; % convert to star
        nodeTrueVals = target(i,:);
        nodeHs = vals2Hs_regression(nodeTrueVals,delta); % helper function below
        res = verify_specification(Y,nodeHs);
        if res == 2
            % check is propery is violated
            res = checkViolated_regression(Y, target(i,:),delta); % helper function below
        end
        results(i) = res;
    end
end

function Hs = vals2Hs_regression(y_ref, delta)
    % Generate 2 halfspaces per output dim:
    % y_i <= y_ref(i) + delta  →  [1 0 ...] * y <= b
    % y_i >= y_ref(i) - delta  →  [-1 0 ...] * y <= -a

    n = length(y_ref);
    G = [ eye(n);    % upper bounds
         -eye(n) ];  % lower bounds

    g = [ y_ref + delta;
         -y_ref + delta ];

    Hs = [];
    for i = 1:size(G,1)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end
end

function res = checkViolated_regression(Set, y_ref, delta)
    res = 2;  % Start assuming unknown

    [lb, ub] = Set.getRanges;

    % Make sure all vectors are column vectors and sizes match
    lb = lb(:);
    ub = ub(:);
    y_ref = y_ref(:);

    if any(lb < (y_ref - delta)) || any(ub > (y_ref + delta))
        res = 0;  % robustness violated
    else
        res = 1;  % robust
    end
end

function parsave(modelPath, epsilon, results, outputSets, rT, targets, timing, rmse, specific_perturbation)
    if specific_perturbation
        fname = "results/gine_sp/verified_nodes_" + modelPath + "_eps" + string(epsilon) + ".mat";
    else
        fname = "results/gine/verified_nodes_" + modelPath + "_eps" + string(epsilon) + ".mat";
    end
    save(fname, "results", "outputSets", "rT", "targets", "timing", "rmse", '-v7.3');
end

function rmse = get_rmse_from_log(bus_system, timestamp)
%GET_RMSE_FROM_LOG Parses training log to get RMSE for a given model.
%   rmse = GET_RMSE_FROM_LOG(bus_system, timestamp)
%   Returns test RMSE as double if found, otherwise NaN.

    rmse = NaN;  % default value
    results_file = fullfile("results", sprintf('%s_gcn_pf_train_%s.txt', bus_system, timestamp));

    if ~exist(results_file, 'file')
        warning('Training log not found: %s', results_file);
        return;
    end

    fid = fopen(results_file, 'r');
    lines = {};
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if ~isempty(line)
            lines{end+1} = line; %#ok<AGROW>
        end
    end
    fclose(fid);

    % Search for the line containing numerical values
    for i = 1:length(lines)
        parts = strsplit(lines{i}, '\t');
        if numel(parts) >= 3 && ~isnan(str2double(parts{3}))
            rmse = str2double(parts{3});
            return;
        end
    end
end
