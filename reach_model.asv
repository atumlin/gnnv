%% Getting the output reachable set of a Graph Neural Network
% This code is specifically looking at PF Analysis using GINEConv Layers
% Author: Anne Tumlin
% Date: 03/15/2025

function reach_model(modelPath,epsilon)
    
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

        for i = 1:numel(X_test)

            X = X_test{i};
            X = dlarray(X);

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
            
            Y = computeReachability({w1,w2,w3}, {ew1,ew2,ew3}, L, reachMethod, Xverify, EVerify, AVerify);

            % store results
            outputSets{i} = Y;
            target_vals = Y_test{1,i};
            targets{i} = target_vals;
            rT{i} = toc(t);
    
         end

         % Save verification results
        [~, baseName, ~] = fileparts(modelPath);  % removes 'models/' and '.mat'
        save("results/verified_nodes_" + baseName + "_eps" + string(epsilon(k)) + ".mat", ...
             "outputSets", "targets", "rT");
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
V = Xverify.V; % 24x4x1x97
c = V(:,:,1,1); % 24x4

% Step 1)  Add features and edge features (adjust c)
% Need to transform features to match dimensionality for addition 
ew1 = extractdata(edge_weights{1}); 
newc = addEdgeNodeFeatures(c,Everify,ew1); % 24x4
V(:,:,1,1) = newc; % 24x4x1x97

% Step 2) Create new star with new c 
X2 = ImageStar(V, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);

% Step 3) Apply ReLU
X3 = L.reach(X2, reachMethod);

% Step 4) Multiply by adjacency 
% newV = X3.V; % 24x4x1x97
newV = squeeze(X3.V); % 24x4x97
Averify_full = full(Averify); % 24x24
newV = tensorprod(Averify_full, newV, 2, 1); % 24x4x97

% Step 5) Add updated features with original features 
newc = newV(:,1:4);
newc = newc + c;
newV(:,1:4) = newc; % 24x4x97

% Step 6) Multiply by weights
W1 = extractdata(weights{1}); % 4x32
newV = tensorprod(newV, W1, 2, 1); % 24x97x32
newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); %24x97x1x32
newV = permute(newV, [1 4 3 2]); % 24x32x1x97

% Step 7) Create new ImageStar
X4 = ImageStar(newV, X3.C, X3.d, X3.pred_lb, X3.pred_ub);

%%%%%%%%% Layer 2 %%%%%%%%%%%
V = X4.V; % 24x32x1x97
c = V(:,:,1,1); % 24x32

ew2 = extractdata(edge_weights{2}); 
newc = addEdgeNodeFeatures(c,Everify,ew2); % 24x32
V(:,:,1,1) = newc; % 24x32x1x97

X5 = ImageStar(V, X4.C, X4.d, X4.pred_lb, X4.pred_ub);

X6 = L.reach(X5, reachMethod);

newV = squeeze(X6.V); % 24x32x97

newV = tensorprod(Averify_full, newV, 2, 1); % 24x32x97

tempc = newV(:,1:32);
tempc = tempc + c;
newV(:,1:32) = tempc; % 24x32x97

W2 = extractdata(weights{2}); % 32x32
newV = tensorprod(newV, W2, 2, 1); % 24x97x32
newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); %24x97x1x32
newV = permute(newV, [1 4 3 2]); % 24x32x1x97

X7 = ImageStar(newV, X6.C, X6.d, X6.pred_lb, X6.pred_ub);

%%%%%%%%% Layer 3 %%%%%%%%%%%

V = X7.V; % 24x32x1x97
c = V(:,:,1,1); % 24x32

ew3 = extractdata(edge_weights{3}); 
newc = addEdgeNodeFeatures(c,Everify,ew3); % 24x32
V(:,:,1,1) = newc; % 24x32x1x97

X8 = ImageStar(V, X7.C, X7.d, X7.pred_lb, X7.pred_ub);

X9 = L.reach(X8, reachMethod);

newV = squeeze(X9.V); % 24x32x97

newV = tensorprod(Averify_full, newV, 2, 1); % 24x32x97

tempc = newV(:,1:32);
tempc = tempc + c;
newV(:,1:32) = tempc; % 24x32x97

W3 = extractdata(weights{3}); % 32x4
newV = tensorprod(newV, W3, 2, 1); % 24x97x4
newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); %24x97x1x4
newV = permute(newV, [1 4 3 2]); % 24x4x1x97

Y = ImageStar(newV, X9.C, X9.d, X9.pred_lb, X9.pred_ub);
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

