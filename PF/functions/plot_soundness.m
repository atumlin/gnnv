%% Make a plot to check that the true prediction falls within the bounds of the reachable set
function plot_soundness(Yreach, model_file, row_idx)
    % plot_soundness - Visualize reachable output ranges vs true predictions
    % 
    % Inputs:
    %   Yreach      - ImageStar object from verification (output reach set)
    %   model_file  - path to .mat file with model and test data
    %   row_idx     - (optional) which row in the output to plot (default = 1)
    %
    % This function generates 4 subplots (one per feature), each showing:
    %   - Reachable bounds (with horizontal bar caps)
    %   - True prediction (as a red dot)

    if nargin < 3
        row_idx = 1;
    end

    %% Load model and data
    model_data = load(model_file);
    parameters = model_data.parameters;
    ANorm = model_data.ANorm;
    E = model_data.E;
    X_test = model_data.X_test;
    Y_test = model_data.Y_test;

    %% Get input and compute true prediction
    X_input = X_test{row_idx};
    Y_true_all = Y_test{row_idx};

    % Forward pass through GNN
    Y_pred_all = GINEConv(ANorm, X_input, E, parameters);  % Nx4 dlarray
    Y_pred_all = extractdata(Y_pred_all);  % Convert to double
    Y_pred_row = Y_pred_all(row_idx, :);   % 1x4

    %% Get reachable output bounds from ImageStar
    [lb, ub] = Yreach.getRanges();  % Nx4
    lb_row = lb(row_idx, :);        % 1x4
    ub_row = ub(row_idx, :);        % 1x4

    %% Plot
    cap_width = 0.1;  % Width of horizontal caps

    figure;
    for i = 1:4
        subplot(1, 4, i);  % 1 row, 4 columns

        x = 1;
        y_lower = lb_row(i);
        y_upper = ub_row(i);
        y_pred = Y_pred_row(i);

        hold on;

        % Vertical line between lower and upper bounds
        plot([x x], [y_lower y_upper], 'b-', 'LineWidth', 2);

        % Horizontal caps
        plot([x - cap_width, x + cap_width], [y_lower y_lower], 'b-', 'LineWidth', 2);
        plot([x - cap_width, x + cap_width], [y_upper y_upper], 'b-', 'LineWidth', 2);

        % True prediction as red dot
        scatter(x, y_pred, 50, 'r', 'filled');

        % Style
        xlim([0.5 1.5]);
        title(['Feature ' num2str(i)]);
        ylabel('Output value');
        set(gca, 'XTick', []);
        grid on;
    end

    sgtitle(sprintf('Reachable Set vs Prediction (row %d)', row_idx), 'FontWeight', 'bold');
end



%% Helper Functions
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