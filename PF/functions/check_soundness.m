%% Make a plot to check that the true prediction falls within the bounds of the reachable set
function check_soundness(Yreach, Y_pred, row_idx)
    % plot_soundness - Visualize reachable output ranges vs true predictions
    % 
    % Inputs:
    %   Yreach      - ImageStar object from verification (output reach set)
    %   Y_pred      - the current prediction 
    %   row_idx     - (optional) which row in the output to plot (default = 1)
    %
    % This function generates 4 subplots (one per feature), each showing:
    %   - Reachable bounds (with horizontal bar caps)
    %   - True prediction (as a red dot)

    if nargin < 3
        row_idx = 1;
    end

    if ~isa(Y_pred, 'double')
        Y_pred = extractdata(Y_pred);
    end
    Y_pred_row = Y_pred(row_idx, :);   % 1x4

    % Get reachable output bounds from ImageStar
    [lb, ub] = Yreach.getRanges();  % Nx4
    lb_row = lb(row_idx, :);        % 1x4
    ub_row = ub(row_idx, :);        % 1x4

    % Plot
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