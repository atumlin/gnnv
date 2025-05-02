%% Create visualizations for computed L_inf results
% Author: Diego Manzanas Lopez
% Edited by: Anne Tumlin (April 2025)

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a molecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently
function process_results(seeds,epsilon)
    eN = length(epsilon);
    % Verify one model at a time
    for m=1:length(seeds)
    
        % get model
        modelPath = "gcn_"+string(seeds(m));
        
        % initialize vars
        molecules = zeros(eN,4); % # robust, #unknown, # not robust/misclassified, # molecules
        atoms = zeros(eN,4);     % # robust, #unknown, # not robust/misclassified, # atoms
        
        for k = 1:eN
            
            % Load data one at a time
            load("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat");
        
            N = length(targets);
            
            for i=1:N
                
                % get result data
                res = results{i};
                n = length(res);
                rb  = sum(res==1); % robust
                unk = sum(res==2); % unknown
                nrb = sum(res==0); % not robust
                
                % molecules
                if rb == n
                    molecules(k,1) = molecules(k,1) + 1;
                elseif unk == n
                    molecules(k,2) = molecules(k,2) + 1;
                elseif nrb == n
                    molecules(k,3) = molecules(k,3) + 1;
                end
                molecules(k,4) = molecules(k,4) + 1;
                
                % atoms
                atoms(k,1) = atoms(k,1) + rb;
                atoms(k,2) = atoms(k,2) + unk;
                atoms(k,3) = atoms(k,3) + nrb;
                atoms(k,4) = atoms(k,4) + n;
        
            end
                
        end
    
        % Save summary
        save("results/summary_results_Linf_"+modelPath+".mat", "atoms", "molecules");
        
        model = load("models/"+modelPath+".mat");
        
        % Create table with these values
        fileID = fopen("results/summay_results_Linf_"+modelPath+".txt",'w');
        fprintf(fileID, 'Summary of robustness results of gnn model with accuracy = %.4f \n\n', model.accuracy);
        
        % Molecule stats
        fprintf(fileID, '               MOLECULES \n');
        fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N     Time (s)\n');
        
        for k = 1:eN
            % Load the corresponding time value
            tmp = load("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat");
            t_elapsed = tmp.time;
        
            fprintf(fileID, ' %7.4f | %.3f    %.3f   %.3f   %d    %.2f\n', ...
                epsilon(k), ...
                molecules(k,1)/molecules(k,4), ...
                molecules(k,2)/molecules(k,4), ...
                molecules(k,3)/molecules(k,4), ...
                molecules(k,4), ...
                t_elapsed);
        end
        
        fprintf(fileID, '\n\n');
        
        % Atom stats
        fprintf(fileID,'                 ATOMS \n');
        fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N     Time (s)\n');
        
        for k = 1:eN
            tmp = load("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat");
            t_elapsed = tmp.time;
        
            fprintf(fileID, ' %7.4f | %.3f    %.3f   %.3f   %d    %.2f\n', ...
                epsilon(k), ...
                atoms(k,1)/atoms(k,4), ...
                atoms(k,2)/atoms(k,4), ...
                atoms(k,3)/atoms(k,4), ...
                atoms(k,4), ...
                t_elapsed);
        end
        
        fclose(fileID);

    
    end

end