%% Training GCNs for Power Flow Analysis Task 
% From PowerGraph:
% We are given a edge_attr.mat, edge_index.mat, X.mat, and a Y_polar.mat file. 

%% Understanding the data 
% dataX = load('data/X.mat');
% cellArray = dataX.Xpf;
% firstCell = cellArray{1};
% disp(firstCell)
% 
% dataY = load('data/Y_polar.mat');
% cellArray2 = dataY.Y_polarpf;
% firstCell = cellArray2{1};
% disp(firstCell)

% We are going to need a couple of matricies for this training
% ANorm: normalized adjacency matrix (N x N)
% Z1: node features (N x F)
% E: edge features (N x N x D)

%% Settings
bus_system = 'ieee39';  % Options: 'ieee24', 'ieee39', etc.

seed_list = [6,7,8,9,10]; 

results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
% Construct the filename incorporating the bus system and timestamp
filename = fullfile(results_folder, sprintf('%s_gcn_pf_train_%s.txt', bus_system, timestamp));
fileID = fopen(filename, 'w');
fprintf(fileID, 'Training Results for %s - %s\n', bus_system, timestamp);
fprintf(fileID, '--------------------------------------\n');
fprintf(fileID, 'Seed\tTest MSE\tTest RMSE\n');

% Testing variables 
visual = false;

for s = 1:length(seed_list)
    %% Processing data 
    
    % Construct the data folder path dynamically
    data_folder = sprintf('%s_data', bus_system);
    
    % Load Data from the correct folder
    edge_attr = load(fullfile(data_folder, 'edge_attr.mat')); % Edge features
    edge_index = load(fullfile(data_folder, 'edge_index.mat')); % Edge index (to make A)
    X = load(fullfile(data_folder, 'X.mat')); % Node features
    Y = load(fullfile(data_folder, 'Y_polar.mat')); % Node labels
    
    % Construct Adjacency Matrix
    A = edgeIndexToAdjacency(edge_index.edge_index, size(X.Xpf{1}, 1));
    if visual, figure; plotGraph(A,'Graph Visualization of Adjacency Matrix'); end
    ANorm = normalizeAdjacency(A); % Might have self loops but to be safe
    if visual, figure; plotGraph(ANorm,'ANorm Graph'); end
    
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
    
    % Split Data Train, Test, Validation
    maxSnapshots = 1000; % Define the maximum number of snapshots to use
    numObs = min(numel(X.Xpf), maxSnapshots); % Use the smaller value between 100 and the total available snapshots
    shuffle_idx = randperm(numObs);
    train_idx = shuffle_idx(1:round(0.8 * numObs));
    val_idx = shuffle_idx(round(0.8 * numObs) + 1:round(0.9 * numObs));
    test_idx = shuffle_idx(round(0.9 * numObs) + 1:end);
    
    % Split Data Into Train, Validation, and Test Sets
    X_train = X.Xpf(train_idx);
    Y_train = Y.Y_polarpf(train_idx);
    X_val = X.Xpf(val_idx);
    Y_val = Y.Y_polarpf(val_idx);
    X_test = X.Xpf(test_idx);
    Y_test = Y.Y_polarpf(test_idx);
    
    %% Model Training 
    
    % Initialize weights
    parameters = struct;
    input_dim = size(X.Xpf{1}, 2);
    hidden_dim = 32;
    output_dim = size(Y.Y_polarpf{1}, 2);
    
    sz = [input_dim hidden_dim];
    num_out = hidden_dim;
    num_in = input_dim;
    parameters.mult1.Weights = initializeGlorot(sz,num_out,num_in,"double");
    
    sz = [hidden_dim hidden_dim];
    num_out = hidden_dim;
    num_in = hidden_dim;
    parameters.mult2.Weights = initializeGlorot(sz,num_out,num_in,"double");
    
    sz = [hidden_dim output_dim];
    num_out = output_dim;
    num_in = input_dim;
    parameters.mult3.Weights = initializeGlorot(sz,num_out,num_in,"double");
    
    % Training Hyperparameters
    num_epochs = 50;
    lr = 0.01;
    epsilon = 0.01;
    val_freq = 10;
    
    monitor = trainingProgressMonitor( ...
        Metrics=["TrainingLoss","ValidationLoss"], ...
        Info="Epoch", ...
        XLabel="Epoch");
    
    groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"])
    
    trailingAvg = [];
    trailingAvgSq = [];
    
    for epoch = 1:num_epochs
        for i = 1:numel(X_train)
            % Extract features and labels for each graph
            X = X_train{i};
            Y = Y_train{i};
    
            [loss,gradients] = dlfeval(@modelLoss,ANorm,X,E,parameters,Y);
    
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,epoch,lr);
        end
    
        % Validation Phase
        val_loss = 0;
        for i = 1:numel(X_val)
            % Extract features and labels for each validation graph
            Xv = X_val{i};
            Yv = Y_val{i};
    
            % Compute validation loss (without gradient computation)
            [vloss,vgradients] = dlfeval(@modelLoss,ANorm,Xv,E,parameters,Yv);
            val_loss = val_loss + vloss;
        end
        val_loss = val_loss / numel(X_val); % Average validation loss
    
        % Record training and validation losses
        recordMetrics(monitor, epoch, TrainingLoss=loss, ValidationLoss=val_loss);
        updateInfo(monitor, Epoch=(epoch + " of " + num_epochs));
    
        % Update progress bar
        monitor.Progress = 100 * (epoch / num_epochs);
    end
    
    % Final Test Evaluation
    test_mse = 0;
    test_rmse = 0;
    for i = 1:numel(X_test)
        Xt = X_test{i};
        Yt = Y_test{i};
        % Compute test loss (without gradient computation)
        [mse_loss,tgradients] = dlfeval(@modelLoss,ANorm,Xt,E,parameters,Yt);
        test_mse = test_mse + mse_loss;
    end
    test_mse = test_mse / numel(X_test); % Average test loss
    test_rmse = sqrt(test_mse);
    % Print results to console
    fprintf('Seed: %d | Test MSE: %f | Test RMSE: %f\n', seed_list(s), test_mse, test_rmse);
    
    % Write results to file
    fprintf(fileID, '%d\t%f\t%f\n', seed_list(s), test_mse, test_rmse);

end

fclose(fileID);
fprintf('Results saved to %s\n', filename);

% Show graph for random test result
randomIdx = randi(numel(X_test));

% Extract data for this index
Xt = X_test{randomIdx}; % Node features
Yt = Y_test{randomIdx}; % True labels

% Compute model predictions
Y_pred = GINEConv(ANorm, Xt, E, parameters);

% Plot the results
plotGraphComparison(ANorm, Yt, Y_pred, 'Voltage Magnitude', 3);
plotGraphComparison(ANorm, Yt, Y_pred, 'Voltage Angle', 4);

%% Helper functions %%
%%%%%%%%%%%%%%%%%%%%%%

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
    Z2 = GINEConvLayer(ANorm, Z1, E, parameters.mult1.Weights);

    % Layer 2
    Z3 = GINEConvLayer(ANorm, Z2, E, parameters.mult2.Weights);

    % Layer 3
    Z4 = GINEConvLayer(ANorm, Z3, E, parameters.mult3.Weights);

    % Final ouput 
    Y = Z4; 
end

function Z_next = GINEConvLayer(ANorm, Z, E, W)
    % Single GINEConv Layer (Vectorized Version)
    % ANorm: normalized adjacency matrix (N x N)
    % Z: node features (N x F)
    % E: edge features (N x N x D)
    % W: weight matrix for the MLP (F -> F_out)

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
    Z_agg = relu(Z_agg); % (N x N x F)

    %I think this would work to first use the nxn matrix to only get the relevant features in the nxnxf
    %Z_agg_sum = pagemtimes(ANorm, Z_agg); % (N x N) * (N x N x F) -> (N x N x F)
    ANorm = full(ANorm);
    Z_agg = full(Z_agg);
    Z_agg_sum = pagemtimes(ANorm, Z_agg);
    
    %then aggregate those edge features that are remaining ie not zeroed out from anorm
    Z_new = sum(Z_agg_sum, 2); % Sum over the second dimension -> (N x 1 x F)
    Z_message = reshape(Z_new, [N, F]); % (N x F)

    epsilon = 0.01;
    
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

function plotGraph(A,graphTitle)
    G = graph(A); % Create a graph object from adjacency matrix
    plot(G, 'NodeLabel', 1:size(A, 1));
    title(graphTitle);
end 

function plotGraphComparison(A, trueValues, predValues, graphTitle, featureIdx)
    A = full(A);
    G = graph(A);

    % Extract only the selected feature (column) from the feature matrices
    trueFeature = trueValues(:, featureIdx);
    predFeature = predValues(:, featureIdx);

    % disp(trueFeature)
    % disp(predFeature)

    % Determine global color limits (same for both subplots)
    globalMin = min([trueFeature(:); predFeature(:)]);
    globalMax = max([trueFeature(:); predFeature(:)]);
    
    globalMin = double(extractdata(globalMin));
    globalMax = double(extractdata(globalMax));

    predFeature = double(extractdata(predFeature));

    figure;
    
    colormap(jet);

    % ---- Predicted Feature Plot ----
    subplot(1,2,1);
    p1 = plot(G, 'NodeLabel', 1:size(A,1), 'MarkerSize', 7);
    title([graphTitle ' - (Predicted) ']);
    colorbar;
    drawnow;
    p1.NodeCData = predFeature; 
    clim([globalMin, globalMax]); 
    
    % ---- True Feature Plot ----
    subplot(1,2,2);
    p1 = plot(G, 'NodeLabel', 1:size(A,1), 'MarkerSize', 7);
    title([graphTitle ' - (True)']);
    colorbar;
    drawnow;
    p1.NodeCData = trueFeature; 
    clim([globalMin, globalMax]); 

end

function out = relu(x)
    out = max(0, x); % Element-wise maximum between 0 and the input
end

function [loss,gradients] = modelLoss(ANorm,X,E,parameters,T)
    Y_pred = GINEConv(ANorm, X, E, parameters);
    loss = mean((Y_pred - T).^2, 'all');
    gradients = dlgradient(loss, parameters);
end
