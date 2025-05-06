%% Getting the output reachable set of a Graph Neural Network
% This code is specifically looking at PF Analysis using GINEConv Layers
% Author: Anne Tumlin
% Date: 03/15/2025

function reach_pf_gine(modelPath,epsilon)
    
    model_data = load(modelPath);
    
    % Extract model weights (and gather from GPU if needed)
    w1 = gather(model_data.parameters.mult1.Weights);
    w2 = gather(model_data.parameters.mult2.Weights);
    w3 = gather(model_data.parameters.mult3.Weights);

    ew1 = gather(model_data.parameters.edge1.Weights);
    ew2 = gather(model_data.parameters.edge2.Weights);
    ew3 = gather(model_data.parameters.edge3.Weights);
    
    % Extract test graph structure and features
    ANorm = model_data.ANorm;
    E = model_data.E;
    X_test = model_data.X_test;
    Y_test = model_data.Y_test;

    % Store resuts
    targets = {};
    outputSets = {};
    rT = {};

    for k = 1:length(epsilon)
        tic;
        for i = 1:1
        % for i = 1:numel(X_test)

            X = X_test{i};
            X = dlarray(X);
            Z = extractdata(X); 

            % Compute per-column ranges
            range_per_col = max(Z) - min(Z);  % 1x4 vector

            % Adjacency matrix does not change
            AVerify = ANorm;

            % Get the input set
            % Create per-feature epsilon perturbation matrix
            scaled_eps = range_per_col .* epsilon;      % 1x4
            eps_matrix = repmat(scaled_eps, size(X,1), 1);

            lb = extractdata(X) - eps_matrix;
            ub = extractdata(X) + eps_matrix;
        
            Xverify = ImageStar(lb, ub);

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
            
            Y = computeReachability({w1,w2,w3}, {ew1,ew2,ew3}, L, reachMethod, Xverify, EVerify, AVerify);

            % store results
            outputSets{i} = Y;
            target_vals = Y_test{1,i};
            targets{i} = target_vals;
            rT{i} = toc(t);
    
        end
        elapsed = toc;
        fprintf("Reachable set computation took %.4f seconds.\n", elapsed);

         % Save verification results
        [~, baseName, ~] = fileparts(modelPath);  % removes 'models/' and '.mat'
        % save("results/gine/verified_nodes_" + baseName + "_eps" + string(epsilon(k)) + ".mat", ...
        %     "outputSets", "targets", "rT", '-v7.3');
        save("results/gine/verified_nodes_" + baseName + ".mat", ...
            "outputSets", "targets", "rT", '-v7.3');

    end
    
end

%% Helper Functions 

function Y = computeReachability(weights,edge_weights,L,reachMethod,input,edgeMat,adjMat)
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
V = Xverify.V; % nodes x features x 1 x terms

% Step 1)  Add features and edge features (adjust c)
% Need to transform features to match dimensionality for addition 
ew1 = extractdata(edge_weights{1}); 
V = addEdgeNodeFeaturesStarSet(V, Everify, ew1);

% Step 2) Create new star with new V
X2 = ImageStar(V, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);

% Step 3) Apply ReLU
X3 = L.reach(X2, reachMethod);

% Step 4) Multiply by adjacency 
newV = squeeze(X3.V); % nodes x features x terms 
Averify_full = full(Averify); % nodes x nodes
newV = tensorprod(Averify_full, newV, 2, 1); % nodes x features x terms 

% Step 5) Add updated features with original features 
sz = size(newV);
newV = reshape(newV, [sz(1), sz(2), 1, sz(3)]);  % nodes × features × 1 × terms
Xagg = ImageStar(newV, X3.C, X3.d, X3.pred_lb, X3.pred_ub);

% Ensure both ImageStars have the same number of predicate variables
n_preds = size(Xagg.V, 4);
Xverify_padded = padStarSetToMatch(Xverify, n_preds);

% Now safe to add
Xskip = Xagg.MinkowskiSum(Xverify_padded);

% Step 7) Apply final weights dynamically
W1 = extractdata(weights{1});  % in dims: [in_feats x out_feats]
newV = tensorprod(Xskip.V, W1, 2, 1);  % Contract over feature dim

% Reshape to 4D for ImageStar: [nodes × terms × 1 × out_feats]
% newV should now be [nodes × 1 x terms × out_feats]
newV = permute(newV, [1 4 2 3]);  % nodes × out_feats × 1 × terms

% Step 8) Final ImageStar
X4 = ImageStar(newV, Xskip.C, Xskip.d, Xskip.pred_lb, Xskip.pred_ub);

%%%%%%%%% Layer 2 %%%%%%%%%%%
V = X4.V;  % nodes x features x 1 x terms

ew2 = extractdata(edge_weights{2}); 

V = addEdgeNodeFeaturesStarSet(V, Everify, ew2);

X5 = ImageStar(V, X4.C, X4.d, X4.pred_lb, X4.pred_ub);

X6 = L.reach(X5, reachMethod);

newV = squeeze(X6.V);  % nodes x features x terms 

newV = tensorprod(Averify_full, newV, 2, 1);  % nodes x features x terms 

sz = size(newV);
newV = reshape(newV, [sz(1), sz(2), 1, sz(3)]);  % nodes × features × 1 × terms
Xagg = ImageStar(newV, X6.C, X6.d, X6.pred_lb, X6.pred_ub);

% Ensure both ImageStars have the same number of predicate variables
n_preds = size(Xagg.V, 4);
X_padded = padStarSetToMatch(X4, n_preds);

% Now safe to add
Xskip = Xagg.MinkowskiSum(X_padded);

W2 = extractdata(weights{2});  % in dims: [in_feats x out_feats]
newV = tensorprod(Xskip.V, W2, 2, 1);  % Contract over feature dim
newV = permute(newV, [1 4 2 3]);  % nodes × out_feats × 1 × terms

X7 = ImageStar(newV, Xskip.C, Xskip.d, Xskip.pred_lb, Xskip.pred_ub);

%%%%%%%%% Layer 3 %%%%%%%%%%%

V = X7.V; % nodes x features x 1 x terms

ew3 = extractdata(edge_weights{3}); 

V = addEdgeNodeFeaturesStarSet(V, Everify, ew3);

X8 = ImageStar(V, X7.C, X7.d, X7.pred_lb, X7.pred_ub);

X9 = L.reach(X8, reachMethod);

newV = squeeze(X9.V); % nodes x features x terms 

newV = tensorprod(Averify_full, newV, 2, 1); % nodes x features x terms 

sz = size(newV);
newV = reshape(newV, [sz(1), sz(2), 1, sz(3)]);  % nodes × features × 1 × terms
Xagg = ImageStar(newV, X9.C, X9.d, X9.pred_lb, X9.pred_ub);

% Ensure both ImageStars have the same number of predicate variables
n_preds = size(Xagg.V, 4);
X_padded = padStarSetToMatch(X7, n_preds);

% Now safe to add
Xskip = Xagg.MinkowskiSum(X_padded);

W3 = extractdata(weights{3});  % in dims: [in_feats x out_feats]
newV = tensorprod(Xskip.V, W3, 2, 1);  % Contract over feature dim
newV = permute(newV, [1 4 2 3]);  % nodes × out_feats × 1 × terms

Y = ImageStar(newV, Xskip.C, Xskip.d, Xskip.pred_lb, Xskip.pred_ub);
end

function Z_out = addEdgeNodeFeatures(Z,E,W_e)
    % Z: node features (N x F)
    % E: edge features (N x N x D)

    [N, F] = size(Z); % Number of nodes (N) and node feature dimension (F)
    D = size(E, 3); % Edge feature dimension (D)

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

function V_out = addEdgeNodeFeaturesStarSet(V_in, E, W_e)
    % V_in: node features from ImageStar.V -> size [N x F x 1 x K]
    % E: edge features [N x N x D]
    % W_e: edge weight matrix [D x F]
    
    [N, F, ~, K] = size(V_in);  % N nodes, F features, K predicate terms
    D = size(E, 3);             % edge feature dimension

    V_out = zeros(N, F, 1, K);  % same size as V_in

    % Pre-transform edges once
    E_transformed = reshape(E, [N*N, D]) * W_e;      % [N^2 x F]
    E_transformed = reshape(E_transformed, [N, N, F]);  % [N x N x F]

    for k = 1:K
        Z = V_in(:,:,1,k);  % [N x F]

        % Expand Z for broadcasting
        Z_expanded = reshape(Z, [N, 1, F]);  % [N x 1 x F]

        % Add edge features to node features
        Z_agg = Z_expanded + E_transformed;  % [N x N x F]

        % Aggregate messages
        Z_out = sum(Z_agg, 2);  % [N x 1 x F]
        V_out(:,:,1,k) = squeeze(Z_out);  % [N x F]
    end
end


function Xpad = padStarSetToMatch(X, targetNumPreds)
    % Pads the predicate terms (4th dim of V) to match targetNumPreds

    V = X.V;  % size: N x F x 1 x K
    [N, F, ~, K] = size(V);

    if K == targetNumPreds
        Xpad = X;
        return;
    elseif K > targetNumPreds
        error("Cannot pad to smaller size. X already has more predicate variables.");
    end

    % Pad V with zeros along the 4th dim
    padV = zeros(N, F, 1, targetNumPreds - K);
    V_new = cat(4, V, padV);  % N x F x 1 x targetNumPreds

    % Pad constraints (C, d) with identity rows for new variables
    C_pad = [X.C, zeros(size(X.C,1), targetNumPreds - K)];
    pred_lb_pad = [X.pred_lb; -ones(targetNumPreds - K, 1)];
    pred_ub_pad = [X.pred_ub; ones(targetNumPreds - K, 1)];

    % Return new padded ImageStar
    Xpad = ImageStar(V_new, C_pad, X.d, pred_lb_pad, pred_ub_pad);

end
