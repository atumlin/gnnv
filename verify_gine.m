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
                results{i} = verifyNode(Y, label);
            end

            % Save results
            % parsave(modelPath, eps, results, rdata.outputSets, rdata.rT, rdata.targets);
        end
    end
end

% Check robustness of every node. 
function results = verifyNode(X, target)
node = size(target,1);
results = 3*ones(node,1); 
    for i=1:node
        matIdx = zeros(1,node);
        matIdx(i) = 1;
        Y = X.affineMap(matIdx, []); % Reduce Imagestar to 1 dimension
        Y = Y.toStar; % convert to star
        nodeTrueVals = target(i,:);
        nodeHs = vals2Hs(nodeTrueVals); % helper function below
        res = verify_specification(Y,nodeHs);
        if res == 2
            % check is propery is violated
            res = checkViolated(Y, target(i,:)); % helper function below
        end
        results(i) = res;
    end
end

function Hs = vals2Hs(trueVals)
    % Convert output target to halfspace for verification
    % @Hs: unsafe/not robust region defined as a HalfSpace

    outSize = 5; % num of classes
    % classes = ["H";"C";"N";"O";"S"];

    switch label
        case 'H'
            target = 1;
        case 'C'
            target = 2;
        case 'N'
            target = 3;
        case 'O'
            target = 4;
        case 'S'
            target = 5;
    end

    % Define HalfSpace Matrix and vector
    G = ones(outSize,1);
    G = diag(G);
    G(target, :) = [];
    G = -G;
    G(:, target) = 1;

    g = zeros(size(G,1),1);

    % Create HalfSapce to define robustness specification
    Hs = [];
    for i=1:length(g)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end

end

function res = checkViolated(Set, label)
    res = 2; % assume unknown (property is not unsat, try to sat)
    % get target label index
    switch label
        case 'H'
            target = 1;
        case 'C'
            target = 2;
        case 'N'
            target = 3;
        case 'O'
            target = 4;
        case 'S'
            target = 5;
    end
    % Get bounds for every index
    [lb,ub] = Set.getRanges;
    maxTarget = ub(target);
    % max value of the target index smaller than any other lower bound?
    if any(lb > maxTarget)
        res = 0; % falsified
    end
end




function parsave(modelPath, epsilon, results, outputSets, rT, targets)
    fname = "results/verified_nodes_" + modelPath + "_eps" + string(epsilon) + ".mat";
    save(fname, "results", "outputSets", "rT", "targets");
end