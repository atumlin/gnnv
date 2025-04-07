% Getting the output reachable set of a Graph Neural Network
% This code is specifically looking at PF Analysis using GINEConv Layers

function reach_model(modelPath,bus_system,max_snapshots,epsilon)
    
    %% Load GCN parameters
    load("models/"+modelPath);
    
    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);
    
    %% Compute reachability, process one bus at a time
    
    [ANorm, E, X_test, Y_test] = get_test_data(bus_system,max_snapshots);

    num_graphs = 100;

    N = size(ANorm,1);

    % Store resuts
    targets = {};
    outputSets = {};
    rT = {};

    for k = 1:length(epsilon)
        for graph = 1:num_graphs
            X = X_test{graph};
            X = dlarray(X);
        for i = 1:N
            
            % Adjacency matrix does not change
            AVerify = ANorm;

            % Get the input set
            lb = extractdata(X-epsilon(k));
            ub = extractdata(X+epsilon(k));
            Xverify = ImageStar(lb,ub);

            % What to do about edge features? 
            % Within the GINEConv layer the edge features are not being
            % transformed or updated. Rather they are being used as an
            % instrument to update the feature matrix. Therefore, we will
            % be considering these as a transformation element but not an
            % entirely new input set that must be transformed for
            % reachability. However, if in next iterations we want to
            % perform perturbations on the edge features, then this will
            % need to be reconsidered.
            EVerify = E;

            % Compute reachability
            t = tic;
            
            reachMethod = 'approx-star';
            L = ReluLayer(); % Create relu layer;
            
            Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, EVerify, AVerify);

            % store results
            outputSets{i} = Y;
            targets{i} = Y_test;
            rT{i} = toc(t);

        end

         % Save verification results
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "outputSets", "targets", "rT");
        end 
    end
    
end

%% Helper Functions 


function [ANorm, E, X_test, Y_test] = get_test_data(bus_system,max_snapshots)
    % Gets data, normalizes, and splits equivalent to training process
    % Inputs:
    % bus_system: Whichever bus_system we want to work with (i.e. 24, 32)
    % maxSnapshots: The number of graphs in our training process should be
    % the number we use here. (Probably 10000)
    % Outputs: 
    % adjacencey matrix, feature matrix, edge matrix, label matrix
    data_folder = sprintf('%s_data', bus_system);
    
    % Load data from the correct folder
    edge_attr = load(fullfile(data_folder, 'edge_attr.mat')); % Edge features
    edge_index = load(fullfile(data_folder, 'edge_index.mat')); % Edge index (to make A)
    X = load(fullfile(data_folder, 'X.mat')); % Node features
    Y = load(fullfile(data_folder, 'Y_polar.mat')); % Node labels
    
    % Construct Adjacency Matrix
    A = edgeIndexToAdjacency(edge_index.edge_index, size(X.Xpf{1}, 1));
    ANorm = normalizeAdjacency(A); % Might have self loops but to be safe
    
    % Normalize Edge Features
    edge_feats = edge_attr.edge_attr;
    E_norm = (edge_feats - mean(edge_feats, 1)) ./ std(edge_feats, 0, 1);
    
    % Construct Edge Feature Matrix - E -> NxNxD
    num_nodes = size(A, 1);
    D = size(E_norm, 2);
    E = zeros(num_nodes, num_nodes, D);
    
    for i = 1:size(edge_index.edge_index, 1)
        src = edge_index.edge_index(i, 2);
        dest = edge_index.edge_index(i, 1);
        E(src, dest, :) = E_norm(i, :);
        E(dest, src, :) = E_norm(i, :);
    end
    
    % Compute global normalization stats
    global_mean = mean(cell2mat(X.Xpf(:)), 1);
    global_std = std(cell2mat(X.Xpf(:)), 0, 1);
    global_mean_labels = mean(cell2mat(Y.Y_polarpf(:)), 1);
    global_std_labels = std(cell2mat(Y.Y_polarpf(:)), 0, 1);
    
    % Normalize each snapshot
    for i = 1:numel(X.Xpf)
        X.Xpf{i} = (X.Xpf{i} - global_mean) ./ global_std;
        Y.Y_polarpf{i} = (Y.Y_polarpf{i} - global_mean_labels) ./ global_std_labels;
    end
    
    % Get the test data 
    numObs = min(numel(X.Xpf), max_snapshots); % Use the smaller value between maxSnapshots and the total available snapshots
    shuffle_idx = randperm(numObs);
    test_idx = shuffle_idx(round(0.9 * numObs) + 1:end);

    X_test = X.Xpf(test_idx);
    Y_test = Y.Y_polarpf(test_idx);
end

function ANorm = normalizeAdjacency(A)
    % Add self connections to adjacency matrix.
    A = A + speye(size(A));
    
    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));
    
    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);
end

function A = edgeIndexToAdjacency(edge_index, num_nodes)
    % Take edge indexes and convert into A matrix
    % Inputs: 
    % edge_index: num. edges x 2 where col 1 is destination node and col 2
    % is source node.
    % num_nodes: N
    A = sparse(edge_index(:,2), edge_index(:,1), 1, num_nodes, num_nodes);
    A = A + A'; % Make symmetric for undirected graphs
end

function Y = computeReachability(weights,L,reachMethod,input,edgeMat,adjMat)
% weights = weights of GNN ({w1, w2, w3}
% L = Layer type (ReLU)
% reachMethod = reachability method for all layers('approx-star is default)
% input = pertubed input features (ImageStar)
% adjMat = adjacency matric of corresonding input features
% edgeMat = 
% Y = computed output of GNN (ImageStar)

Xverify = input; % ImageStar
Averify = adjMat; % N x N 
Everify = edgeMat; % N x N x E
n = size(adjMat,1);

%%%%%%%%% Layer 1 %%%%%%%%%%%
% H_k = (H_(k-1) + A*ReLU(H_(k-1)+E))W_k

% This is the transformation of c and V vectors where c is V0 and V1...Vf
% is a diagonal matrix with the perturbations 
V = Xverify.V;
c = V(:,:,1,1); % N x F 
% Step 1)  Add features and edge features (adjust c)
% Need to transform features to match dimensionality for addition 
newc = addEdgeNodeFeatures(c,Everify); % N x F 
V(:,:,1,1) = newc;
% Step 2) Create new star with new c 
X2 = ImageStar(V, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
% Step 3) Apply ReLU
X2b = L.reach(X2, reachMethod);
X3 = X2b.MinkowskiSum(Xverify);
% Step 4) Multiply by adjacency 
newV = X3.V; % 24x4x1x97
newV = reshape(newV, 24, []); % 4D double to n x (numel(A)*features+features) (24 x 388)
newV = Averify * newV; % (24 x 388)
% Step 5) 


Y = 0;


end

function Z_out = addEdgeNodeFeatures(Z,E)
    % Z: node features (N x F)
    % E: edge features (N x N x D)

    [N, F] = size(Z); % Number of nodes (N) and node feature dimension (F)
    D = size(E, 3); % Edge feature dimension (D)
    W_e = initializeWeights(D, F);

    %W_e is linear transform to get edge features in same dim as node feats
    E_transformed = reshape(E, [N*N, D]) * W_e; % (N^2 x D) * (D x F) -> (N^2 x F)

    %bring it back to make more effecient addition of xj + eji (from the eq in fig above)
    E_transformed = reshape(E_transformed, [N, N, F]); % Back to (N x N x F)

    %convert NxF into Nx1xF (seen better in image below)
    Z_expanded = reshape(Z, [N, 1, F]); % Reshape to (N x 1 x F) for broadcasting

    %do the addition and relu so now we have the features associated with every edge
    Z_agg = Z_expanded + E_transformed; % (N x 1 x F) + (N x N x F) -> (N x N x F)

    Z_out = sum(Z_agg, 2); % N x 1 x F
    Z_out = squeeze(Z_out); % N x F 
end 

function W = initializeWeights(input_dim, output_dim)
    W = randn(input_dim, output_dim) * sqrt(2 / (input_dim + output_dim));
end