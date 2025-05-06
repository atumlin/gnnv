function check_sound(Yreach, Y_pred)
    % check_soundness - Verifies if predictions are within reachable bounds
    %
    % Inputs:
    %   Yreach  - ImageStar output reach set
    %   Y_pred  - Predicted output (dlarray or double), NxF
    %
    % Output:
    %   Prints which rows/features are sound or violated

    % Convert prediction to double if needed
    if ~isa(Y_pred, 'double')
        Y_pred = extractdata(Y_pred);
    end

    [lb, ub] = Yreach.getRanges();  % NxF
    N = size(Y_pred, 1);
    F = size(Y_pred, 2);

    all_sound = true;

    for i = 1:N
        for j = 1:F
            y = Y_pred(i, j);
            l = lb(i, j);
            u = ub(i, j);

            if y < l || y > u
                fprintf("Violation at row %d, feature %d: pred = %.4f, bounds = [%.4f, %.4f]\n", ...
                    i, j, y, l, u);
                all_sound = false;
            end
        end
    end

    if all_sound
        fprintf("All predictions are sound (within reachable bounds).\n");
    end
end
