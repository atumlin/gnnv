%% Verify the robustness of GINEConv models 
% This code is specifically looking at PF Analysis using GINEConv Layers.
% This is going to be a regression task. 
% Author: Anne Tumlin
% Date: 04/02/2025

function verify_gine(epsilons, models)
    for m = 1:length(models)
        modelPath = models(m);

        for k = 1:length(epsilons)
            eps = epsilons(k);
            % Load results
            rdata = load("results/verified_nodes_" + modelPath + "_eps" + string(eps) + ".mat");

            % Per-node verification
            results = cell(length(rdata.outputSets), 1);
            for i = 1:length(rdata.outputSets)
                Y = rdata.outputSets{i};
                label = rdata.targets{i};
                results{i} = verifyNode(Y, label, eps);
            end

            % Save results
            % parsave(modelPath, eps, results, rdata.outputSets, rdata.rT, rdata.targets);
        end
    end
end

% Check robustness of every node. 
function results = verifyNode(X, target,epsilon)
node = size(target,1);
results = 3*ones(node,1); 
    for i=1:node
        matIdx = zeros(1,node);
        matIdx(i) = 1;
        Y = X.affineMap(matIdx, []); % Reduce Imagestar to 1 dimension
        Y = Y.toStar; % convert to star
        nodeTrueVals = target(i,:);
        nodeHs = vals2Hs_regression(nodeTrueVals,epsilon); % helper function below
        res = verify_specification(Y,nodeHs);
        if res == 2
            % check is propery is violated
            res = checkViolated_regression(Y, target(i,:),epsilon); % helper function below
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


function parsave(modelPath, epsilon, results, outputSets, rT, targets)
    fname = "results/verified_nodes_" + modelPath + "_eps" + string(epsilon) + ".mat";
    save(fname, "results", "outputSets", "rT", "targets");
end