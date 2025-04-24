%% Load model and test data
model_file = 'models/gcn_ieee24_2025-04-23_13-09-27.mat';  % <-- your actual file
model_data = load(model_file);
    
parameters = model_data.parameters;
ANorm = model_data.ANorm;
E = model_data.E;
X_test = model_data.X_test;
Y_test = model_data.Y_test;
global_std = model_data.global_std;
global_mean = model_data.global_mean;
global_std_labels = model_data.global_std_labels;
global_mean_labels = model_data.global_mean_labels;

idx = 1;
X_input = X_test{idx};
Y_true = Y_test{idx};

%% Predict
Y_pred = GINEConv(ANorm, X_input, E, parameters); 

%% Denormalize prediction
Y_pred = Y_pred .* global_std_labels + global_mean_labels;
Y_true = Y_true .* global_std_labels + global_mean_labels;

%% Output
disp('Predicted Output (node x feature):');
disp(Y_pred);

disp('True Output (node x feature):');
disp(Y_true);

% % Optionally, compare specific features:
% fprintf('\nVoltage Magnitude Prediction (Column 3):\n');
% disp(Y_pred(:,3));


function A = edgeIndexToAdjacency(edge_index, num_nodes)
    % Take edge indexes and convert into A matrix
    % Inputs: 
    % edge_index: num. edges x 2 where col 1 is destination node and col 2
    % is source node.
    % num_nodes: N
    A = sparse(edge_index(:,2), edge_index(:,1), 1, num_nodes, num_nodes);
    A = A + A'; % Make symmetric for undirected graphs
end

function W = initializeWeights(input_dim, output_dim)
    W = randn(input_dim, output_dim) * sqrt(2 / (input_dim + output_dim));
end

function Y = GINEConv(ANorm, Z1, E, parameters)
    % Inputs:
    % ANorm: normalized adjacency matrix (N x N)
    % Z1: node features (N x F) [initial input features]
    % E: edge features (N x N x D) [initial edge features]
    % W1, W2, W3: weight matrices for the MLP at each layer (F -> F_out)

    % Layer 1
    Z2 = GINEConvLayer(ANorm, Z1, E, parameters.mult1.Weights, parameters.edge1.Weights);

    % Layer 2
    Z3 = GINEConvLayer(ANorm, Z2, E, parameters.mult2.Weights, parameters.edge2.Weights);

    % Layer 3
    Z4 = GINEConvLayer(ANorm, Z3, E, parameters.mult3.Weights, parameters.edge3.Weights);

    % Final output 
    Y = Z4; 
end

function Z_next = GINEConvLayer(ANorm, Z, E, W, W_e)
    % Single GINEConv Layer (Vectorized Version)
    % ANorm: normalized adjacency matrix (N x N)
    % Z: node features (N x F)
    % E: edge features (N x N x D)
    % W: weight matrix for the MLP (F -> F_out)

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


    Z_agg = sum(Z_agg, 2); % N x 1 x F
    Z_agg = squeeze(Z_agg); % N x F 

    Z_agg = relu(Z_agg); % (N x F)

    %Z_agg_sum = pagemtimes(ANorm, Z_agg); % (N x N) * (N x F) -> (N x F)
    ANorm = full(ANorm);
    Z_agg = full(Z_agg);
    Z_message = pagemtimes(ANorm, Z_agg);
    
    % %then aggregate those edge features that are remaining ie not zeroed out from anorm
    % Z_new = sum(Z_agg_sum, 2); % Sum over the second dimension -> (N x 1 x F)
    % Z_message = reshape(Z_new, [N, F]); % (N x F)

    epsilon = 0.00;
    
    %as in GINE it combines that Z message with the original xi
    Z_next = (1 + epsilon) * Z + Z_message;

    %apply the final htheta transformation from figure above
    Z_next = Z_next * W; % (N x F) * (F x F_out) -> (N x F_out)
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

% ReLU implementation 
function out = relu(x)
    out = max(0, x); % Element-wise maximum between 0 and the input
end