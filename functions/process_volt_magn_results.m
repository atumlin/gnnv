% Process verification results for voltage magnitude safety and write 
% summaries to results/safety directory.

function process_volt_magn_results(epsilons,models,specific_perturbation)

    for m = 1:length(models)
        model_path = models(m);
        eN = length(epsilons);
    
        robust_counts    = zeros(eN,1);
        unknown_counts   = zeros(eN,1);
        notrobust_counts = zeros(eN,1);
        total_outputs    = zeros(eN,1);
        timings          = NaN(eN,1);
        rmse_value       = NaN;
    
        for k = 1:eN
            eps = epsilons(k);
            if specific_perturbation
                result_file = "results/gine_sp/volt_magn_results/verified_nodes_" + model_path + "_eps" + string(eps) + ".mat";
            else
                result_file = "results/gine/volt_magn_results/verified_nodes_" + model_path + "_eps" + string(eps) + ".mat";
            end
    
            if exist(result_file, 'file') == 2
                data = load(result_file);
                results = data.results;
    
                for i = 1:length(results)
                    res = results{i};
                    robust_counts(k)    = robust_counts(k)    + sum(res == 1);
                    unknown_counts(k)   = unknown_counts(k)   + sum(res == 2);
                    notrobust_counts(k) = notrobust_counts(k) + sum(res == 0);
                    total_outputs(k)    = total_outputs(k)    + numel(res);
                end
    
                timings(k) = data.timing;
                if isfield(data, 'rmse')
                    rmse_value = data.rmse;
                end
            else
                warning("Missing result file: %s", result_file);
            end
        end

        if specific_perturbation
            % Save summary
            save("results/gine_sp/volt_magn_results/summary_" + model_path + ".mat", ...
                "robust_counts", "unknown_counts", "notrobust_counts", ...
                "total_outputs", "timings");
            % Write text summary
            summary_txt = "results/gine_sp/volt_magn_results/summary_" + model_path + ".txt";
            fileID = fopen(summary_txt, 'w');
            fprintf(fileID, 'Voltage Magnitude Safety Summary for Model w/ SP: %s\n\n', model_path);
            fprintf(fileID, 'Model RMSE: %.4f\n\n', rmse_value);
            fprintf(fileID, 'Epsilon | Robust   Unknown   NotRobust   Total    Time (sec)\n');
            for k = 1:eN
                if total_outputs(k) > 0
                    fprintf(fileID, ' %.4f  |  %.3f    %.3f     %.3f     %d     %.2f\n', ...
                        epsilons(k), ...
                        robust_counts(k)/total_outputs(k), ...
                        unknown_counts(k)/total_outputs(k), ...
                        notrobust_counts(k)/total_outputs(k), ...
                        total_outputs(k), ...
                        timings(k));
                else
                    fprintf(fileID, ' %.4f  |   N/A      N/A       N/A       0     N/A\n', epsilons(k));
                end
            end
            fclose(fileID);
        else
            % Save summary
            save("results/gine/volt_magn_results/summary_" + model_path + ".mat", ...
                "robust_counts", "unknown_counts", "notrobust_counts", ...
                "total_outputs", "timings");
            % Write text summary
            summary_txt = "results/gine/volt_magn_results/summary_" + model_path + ".txt";
            fileID = fopen(summary_txt, 'w');
            fprintf(fileID, 'Voltage Magnitude Safety Summary for Model: %s\n\n', model_path);
            fprintf(fileID, 'Model RMSE: %.4f\n\n', rmse_value);
            fprintf(fileID, 'Epsilon | Robust   Unknown   NotRobust   Total    Time (sec)\n');
            for k = 1:eN
                if total_outputs(k) > 0
                    fprintf(fileID, ' %.4f  |  %.3f    %.3f     %.3f     %d     %.2f\n', ...
                        epsilons(k), ...
                        robust_counts(k)/total_outputs(k), ...
                        unknown_counts(k)/total_outputs(k), ...
                        notrobust_counts(k)/total_outputs(k), ...
                        total_outputs(k), ...
                        timings(k));
                else
                    fprintf(fileID, ' %.4f  |   N/A      N/A       N/A       0     N/A\n', epsilons(k));
                end
            end
            fclose(fileID);

        end
    end
end

