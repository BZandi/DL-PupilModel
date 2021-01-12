%% Description:
% Demo code from the article:
% Deep learning based pupil model predicts time and wavelength dependent light responses
% Technical University of Darmstadt, Laboratory of Lighting Technology
% Published in Scientific Reports
% Link: www.nature.com/articles/s41598-020-79908-5
% GitHub Link: https://github.com/BZandi/DL-PupilModel

function [out, PupilModelResult] = NNCombinedModel(hparam)
    
    %  -> CHANGE: Please support a path to your python env
    python_path = '/Users/papillon/opt/anaconda3/envs/ML/bin/python3';
    
    % Unity-based normalisation --------
    Func_Normalisation = @(x, min_x, max_x) (x - min_x)/(max_x - min_x);
    Func_NormalisationToValues = @(Z, min_x, max_x) Z.*max_x-Z.*min_x+min_x;
    % -----------------------------------
    
    if strcmp(hparam.Condition, 'Single')
        
        % Constants to normalize input and output of NN:
        % Luminance, CIEx, CIEy, Rod_Signal, S_Signal, M_Signal, L_Signal, Melanopsin_Signal
        % f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        max_xV = [100.170000000000 0.717010000000000 0.739280000000000 1.48613000000000 3.24582000000000,...
            0.483380000000000 0.291300000000000 1.81725000000000 20 6.19300000000000 -0.824000000000000,...
            0.487800000000000 1.85000000000000 0.134200000000000 0.869600000000000 2.38685423130276e-19,...
            -2.38905680294312e-17 2.02060208899617e-13 -5.55926435436703e-12 1.30671705091018e-08,...
            -1.52842006068615e-07 0.000113658588265994 -0.000612094739840875 0.110806416713225 2.70547744398636];
        
        % Constants to normalize input and output of NN
        % Luminance, CIEx, CIEy, Rod_Signal, S_Signal, M_Signal, L_Signal, Melanopsin_Signal
        % f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        min_xV = [99.7300000000000 0.158110000000000 0.0200600000000000 0.00232000000000000,...
            9.00000000000000e-05 0.0236300000000000 0.140910000000000 0.000750000000000000,...
            -0.649200000000000 -1.19000000000000 -0.880000000000000 0.100000000000000 0.430600000000000,...
            0.0604000000000000 0.0878000000000000 1.51215164972616e-20 -3.37931381239379e-16 1.56563570647668e-14,...
            -6.63968897519939e-11 1.17384511647451e-09 -1.57360392626362e-06 1.23830930427218e-05 -0.00464651170544400,...
            0.0123988673156660 1.92268530982973];
        
        switch hparam.Variant
            case 1 % Input parameter: Luminance, CIEx, CIEy
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(1), max_xV(1)); % L
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(2), max_xV(2)); % CIEx
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(3), max_xV(3)); % CIEy
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --L',{' '}, num2str(paramValues(1)),...
                    ' --Fx',{' '}, num2str(paramValues(2)),...
                    ' --Fy',{' '}, num2str(paramValues(3)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
                
            case 2 % Input parameter: Lcone, Mcone, Scone, Mel
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(7), max_xV(7)); % Lcone
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(6), max_xV(6)); % Mcone
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(5), max_xV(5)); % Scone
                paramValues(4) = Func_Normalisation(hparam.Stimuli(4), min_xV(8), max_xV(8)); % Melanopsin_Signal
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --Lcone',{' '}, num2str(paramValues(1)),...
                    ' --Mcone',{' '}, num2str(paramValues(2)),...
                    ' --Scone',{' '}, num2str(paramValues(3)),...
                    ' --Mel',{' '}, num2str(paramValues(4)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
                
            case 3 % Input parameter: Luminance, CIEx, CIEy, Mel
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(1), max_xV(1)); % L
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(2), max_xV(2)); % CIEx
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(3), max_xV(3)); % CIEy
                paramValues(4) = Func_Normalisation(hparam.Stimuli(4), min_xV(8), max_xV(8)); % Melanopsin_Signal
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --L',{' '}, num2str(paramValues(1)),...
                    ' --Fx',{' '}, num2str(paramValues(2)),...
                    ' --Fy',{' '}, num2str(paramValues(3)),...
                    ' --Mel',{' '}, num2str(paramValues(4)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
        end
        
        % Read out the predicted results from NN
        Predicted = readtable('Python/output.csv');
        
        % Round the predicted values from the neural network
        Predicted = varfun(@(var) round(var, 6), Predicted);
        
        % Transform the normalized output from the neural network back to absolute
        for Col = 1:size(Predicted, 2)
            InputValuesModel(1, Col) = Func_NormalisationToValues(Predicted{1, Col}, min_xV(Col + 8), max_xV(Col + 8));
        end
        
        % Call the Combined model with the values from the neural network
        PupilModelResult = CombinedPupilModel([0:0.01:300], 33, 199.45, 53.1, 2, 'Single',...
            InputValuesModel(1), InputValuesModel(2), InputValuesModel(3), InputValuesModel(4), InputValuesModel(5),...
            InputValuesModel(6), InputValuesModel(7), InputValuesModel(8), InputValuesModel(9), InputValuesModel(10),...
            InputValuesModel(11), InputValuesModel(12), InputValuesModel(13), InputValuesModel(14), InputValuesModel(15), InputValuesModel(16), InputValuesModel(17));
        % -------------------------------------------------------------------------
        
    elseif strcmp(hparam.Condition, 'Multi')
        
        % Constants to normalize input and output of NN:
        % Luminance, CIEx, CIEy, Rod_Signal, S_Signal, M_Signal, L_Signal, Melanopsin_Signal
        % f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        max_xV = [100.170000000000 0.717010000000000 0.739280000000000 1.48613000000000 3.24582000000000 0.483380000000000 0.291300000000000,...
            1.81725000000000 19.2070000000000 6.19300000000000 -0.824000000000000 0.519700000000000 1.99250000000000 0.134200000000000,...
            0.869600000000000 1.97550116389431e-19 5.80847574168151e-16 1.69958207976320e-13 7.87244804975760e-11 1.13565736212557e-08,...
            8.37511104767027e-07 0.000104951287925765 -0.000538199220672352 0.0968826645369810 2.53009809335703];
        
        % Constants to normalize input and output of NN
        % Luminance, CIEx, CIEy, Rod_Signal, S_Signal, M_Signal, L_Signal, Melanopsin_Signal
        % f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        min_xV = [99.7300000000000 0.158110000000000 0.0200600000000000 0.00232000000000000 9.00000000000000e-05 0.0236300000000000,...
            0.140910000000000 0.000750000000000000 -0.649200000000000 -1.19000000000000 -0.880000000000000 0.100000000000000,...
            0.733500000000000 0.0604000000000000 0 -4.61470419153883e-19 -2.81476800948458e-16 -2.97014221879705e-13 -5.66256848966639e-11,...
            -1.13638105654085e-08 -1.40317255861149e-06 -2.15433976320967e-05 -0.00447454985355800 0.0152416115613120 1.74976367312382];
        
        switch hparam.Variant
            case 1 % Input parameter: Luminance, CIEx, CIEy
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(1), max_xV(1)); % L
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(2), max_xV(2)); % CIEx
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(3), max_xV(3)); % CIEy
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --L',{' '}, num2str(paramValues(1)),...
                    ' --Fx',{' '}, num2str(paramValues(2)),...
                    ' --Fy',{' '}, num2str(paramValues(3)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
                
            case 2 % Input parameter: Lcone, Mcone, Scone, Mel
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(7), max_xV(7)); % Lcone
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(6), max_xV(6)); % Mcone
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(5), max_xV(5)); % Scone
                paramValues(4) = Func_Normalisation(hparam.Stimuli(4), min_xV(8), max_xV(8)); % Melanopsin_Signal
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --Lcone',{' '}, num2str(paramValues(1)),...
                    ' --Mcone',{' '}, num2str(paramValues(2)),...
                    ' --Scone',{' '}, num2str(paramValues(3)),...
                    ' --Mel',{' '}, num2str(paramValues(4)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
                
            case 3 % Input parameter: Luminance, CIEx, CIEy, Mel
                % ----------- Neural Network: Call Python -----------------------------------------------------------
                % Normalize input before feeding into NN
                paramValues(1) = Func_Normalisation(hparam.Stimuli(1), min_xV(1), max_xV(1)); % L
                paramValues(2) = Func_Normalisation(hparam.Stimuli(2), min_xV(2), max_xV(2)); % CIEx
                paramValues(3) = Func_Normalisation(hparam.Stimuli(3), min_xV(3), max_xV(3)); % CIEy
                paramValues(4) = Func_Normalisation(hparam.Stimuli(4), min_xV(8), max_xV(8)); % Melanopsin_Signal
                
                params = strcat('--Condition', {' '}, hparam.Condition,...
                    ' --Variant ',{' '},  num2str(hparam.Variant),...
                    ' --L',{' '}, num2str(paramValues(1)),...
                    ' --Fx',{' '}, num2str(paramValues(2)),...
                    ' --Fy',{' '}, num2str(paramValues(3)),...
                    ' --Mel',{' '}, num2str(paramValues(4)));
                current_path = pwd;
                optimizer_path = strcat(current_path,'/Python/CalcParam.py');
                call_python = char(strcat(python_path, {' '}, optimizer_path, {' '}, params));
                cd('Python/');
                [status, out] = system(call_python);
                cd('../');
        end
        
        % Read out the predicted results from NN
        Predicted = readtable('Python/output.csv');
        
        % Round the predicted values from the neural network
        Predicted = varfun(@(var) round(var, 6), Predicted);
        
        % Transform the normalized output from the neural network back to absolute
        for Col = 1:size(Predicted, 2)
            InputValuesModel(1, Col) = Func_NormalisationToValues(Predicted{1, Col}, min_xV(Col + 8), max_xV(Col + 8));
        end
        
        % call the Combined model with the values from the neural network
        PupilModelResult = CombinedPupilModel([0:0.01:300], 22.1, 199.45, 53.1, 2, 'Multi',...
            InputValuesModel(1), InputValuesModel(2), InputValuesModel(3), InputValuesModel(4), InputValuesModel(5),...
            InputValuesModel(6), InputValuesModel(7), InputValuesModel(8), InputValuesModel(9), InputValuesModel(10),...
            InputValuesModel(11), InputValuesModel(12), InputValuesModel(13), InputValuesModel(14), InputValuesModel(15), InputValuesModel(16), InputValuesModel(17));
        
    end
    
end

function [Results] = CombinedPupilModel(t, ageY, luminanceCDm2, fieldDiameterDEG, eyeNumber, Condition,...
        f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
    
    % Calculated paramter Fan & Yao (2011) model [450nm, 530nm, 610nm, 660nm, 2000K, 5000K, 10000K] ---
    L_0d = [3.3403, 3.3403, 3.3403, 3.3403, 3.3403, 3.3403, 3.3403];        % Constant
    l_0c = [1.0710, 1.0710, 1.0710, 1.0710, 1.0710, 1.0710, 1.0710];        % Constant
    K_d = [1.0714, 1.0714, 1.0714, 1.0714, 1.0714, 1.0714, 1.0714];         % Constant
    K_c = [0, 0, 0, 0, 0, 0, 0];                                            % Constant
    D = [3.4855, 3.4855, 3.4855, 3.4855, 3.4855, 3.4855, 3.4855];           % Constant
    f_p_i_0 = [0, 0, 0, 0, 0, 0, 0];                                        % Constant
    f_s_i_0 = [0, 0, 0, 0, 0, 0, 0];                                        % Constant
    x0_2 = [0, 0, 0, 0, 0, 0, 0];                                           % Constant
    % ----------------------------------------------------------------------------------------------
    
    % Parameter masc function [450nm, 530nm, 610nm, 660nm, 2000K, 5000K, 10000K] ---
    % Funktion: Function_tanh_1 = @(x) 1-(0.5 + 0.5 * tanh((x-q)/r));
    % Funktion: Function_tanh_2 = @(x) 0.5 + 0.5 * tanh((x-q)/r);
    % Masc parameter are different between Single & Multi
    switch Condition
        case 'Single'
            q = [1.5524, 1.5524, 1.5524, 1.5524, 1.5524, 1.5524, 1.5524];                       % Constant
            r = [0.0248, 0.0248, 0.0248, 0.0248, 0.0248, 0.0248, 0.0248];                       % Constant
        case 'Multi'
            q = [1.1359, 1.1359, 1.1359, 1.1359, 1.1359, 1.1359, 1.1359];                       % Constant
            r = [0.3517, 0.3517, 0.3517, 0.3517, 0.3517, 0.3517, 0.3517];                       % Constant
    end
    % -----------------------------------------------------------
    
    % Get inital pupil diameter from offset corrected Watson & Yellot (2012) pupil model
    InitalPupilDiameter = getPupilSizeWatson(ageY, luminanceCDm2, fieldDiameterDEG, eyeNumber, true, Condition);
    
    % Get the phasic time dependent pupil diameter from Fan & Yao (2011) model
    PhasicPupilDiameter = Phasic_Model(L_0d(1), l_0c(1), K_d(1), K_c(1), D(1), f_p, f_s, P_0,...
        tp, ts, f_p_i_0(1), f_s_i_0(1), InitalPupilDiameter, x0_2(1), Delta_tp, Delta_ts, t);
    
    % Get the tonic time dependent pupil diameter from a custom made equation
    TonicPupilDiameter = TonicModel(t, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)';
    
    % Combine the phasic and the tonic model with a masc function
    [G1, G2] = MascFunction(t, q(1), r(1));
    
    CombinedModel = TonicPupilDiameter.*G2(:,2) + PhasicPupilDiameter.*G1;
    
    Results = table(t', PhasicPupilDiameter, TonicPupilDiameter, CombinedModel, 'VariableNames',...
        {'Time' 'PhasicModel' 'TonicModel' 'CombinedModel'});
end

% Offset corrected Watson & Yellot pupil model
function pupilDiameterWatson = getPupilSizeWatson(ageY, luminanceCDm2, fieldDiameterDEG, eyeNumber, Offset, Condition)
    fieldSizeDeg2 = ((fieldDiameterDEG/2)^2)*pi;
    ReferenceAge = 28.58;
    if eyeNumber == 1
        e = 0.1;
    else
        e = 1;
    end
    Function_1 = @(x) 7.75-5.75*(((fieldSizeDeg2*x*e/846)^0.41)/((fieldSizeDeg2*x*e/846)^0.41 +2));
    ResultFunction_1 = Function_1(luminanceCDm2);
    ResultA = 0.021323 - (0.0095623*ResultFunction_1);
    ResultB = (ageY - ReferenceAge) * ResultA;
    
    if Offset == false
        pupilDiameterWatson = ResultFunction_1 + ResultB;
    else
        switch Condition
            case 'Single'
                Offset_Correction = 0.25898;
                pupilDiameterWatson = ResultFunction_1 + ResultB - Offset_Correction;
            case 'Multi'
                Offset_Correction = 0.41834;
                pupilDiameterWatson = ResultFunction_1 + ResultB - Offset_Correction;
        end
    end
end

% Differential equation from Fan & Yao 2011
function Result = Phasic_Model(L_0d, l_0c, K_d, K_c, D, f_p, f_s, P_0,...
        tp, ts, f_p_i_0, f_s_i_0, x0_1, x0_2, Delta_tp, Delta_ts, t)
    
    % Differential equation from Fan & Yao 2011 ------------------------------------
    % Eingaben:
    %   L_0d     -> [mm] Länge des Dilatatormuskels ohne Reiz
    %   l_0c     -> [mm] Radius des Konstriktormuskels ohne Reiz
    %   K_d      -> [mN/g mm^2] Stimulusunabhängige Konstante
    %   K_c      -> [mN/g mm^2] Stimulusunabhängige Konstante
    %   D        -> [mN/g mm] Viskositätskonstante
    %   f_p      -> [mN/g] Zeitabhängige Muskelkraft vom parasympathikus
    %   f_s      -> [mN/g] Zeitabhängige Muskelkraft vom sympathikus
    %   P_0      -> [mN/g] Statische Kraft der Irismuskel
    %   tp       -> [s] Verzögerungsparameter parasympathikus
    %   ts       -> [s] Verzögerungsparameter sympathikus
    %   f_p_i_0  -> [mN/g] Irismuskelkradt parasympathikus ohne Reiz
    %   f_s_i_0  -> [mN/g] Irismuskelkradt sympathikus ohne Reiz
    %   x0_1     -> [mm] Lösung der Differentialgleichung Initialpupillendurchmesser
    %   x0_2     -> [mm] Lösung der Differentialgleichung Initialgeschwindigkeit
    %   Delta_tp -> [] Einschaltzeit des Reizes so wie es aussieht
    % 	Delta_ts -> [] Einschaltzeit des Reizes so wie es aussieht
    % -----------------------------------------------------------------------------------------------------
    
    x0 = [x0_1, x0_2];
    
    [T, Sv] = ode45(@ddefunc, t, x0);
    Result = Sv(:,1);
    
    function [yp] = ddefunc(t, y)
        
        if t < tp || t > tp + Delta_tp
            f_p_i_g = f_p_i_0;
        end
        
        if t >= tp && t <= tp + Delta_tp
            f_p_i = f_p;
            f_p_i_g = f_p_i+ f_p_i_0;
        end
        
        if t < ts || t >  ts + Delta_ts
            f_s_i_g = f_s_i_0;
        end
        
        if t >= ts && t <= ts + Delta_ts
            f_s_i = f_s;
            f_s_i_g = f_s_i+ f_s_i_0;
        end
        
        yp = [y(2);...
            (-K_c*(l_0c - y(1))^(2))+(K_d*(L_0d - y(1))^2)-(D*y(2)) - f_p_i_g + f_s_i_g + P_0];
    end
end

function Result = TonicModel(t, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
    
    Model = @(x) p1*x.^(9) + p2*x.^(8) + p3*x.^(7) + p4*x.^(6) + p5*x.^(5) + p6*x.^(4) + p7*x.^(3) + p8*x.^(2) + p9*x + p10;
    Result = Model(t);
    
end

function [G1, G2] = MascFunction(t, q, r)
    Function_tanh_1 = @(x) 1-(0.5 + 0.5 * tanh((x-q)/r));
    Function_tanh_2 = @(x) 0.5 + 0.5 * tanh((x-q)/r);
    
    g_1 = Function_tanh_1(t);
    g_2 = Function_tanh_2(t);
    
    G1(:,1) = g_1;
    G2(:,2) = g_2;
end