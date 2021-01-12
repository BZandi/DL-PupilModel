%% Description:
% Demo code from the article:
% Deep learning based pupil model predicts time and wavelength dependent light responses
% Technical University of Darmstadt, Laboratory of Lighting Technology
% Published in Scientific Reports
% Link: www.nature.com/articles/s41598-020-79908-5
% GitHub Link: https://github.com/BZandi/DL-PupilModel

    function Result = Plot_DGL_Equation_1(L_0d, l_0c, K_d, K_c, D, f_p, f_s, P_0,...
            tp, ts, f_p_i_0, f_s_i_0, x0_1, x0_2, Delta_tp, Delta_ts, t)
        
        % Das ist die Differentialgleichung aus dem Paper Fan & Yao 2011 ------------------------------------
        % Eingaben:
        %   L_0d     -> [mm] Länge des Dilatatormuskels ohne Reiz                        ---> 3.57 bis 3.59
        %   l_0c     -> [mm] Radius des Konstriktormuskels ohne Reiz                     ---> 0.91 bis 0.93
        %   K_d      -> [mN/g mm^2] Stimulusunabhängige Konstante                        ---> 1.24 bis 1.29
        %   K_c      -> [mN/g mm^2] Stimulusunabhängige Konstante                        ---> 0.047 bis 0.065
        %   D        -> [mN/g mm] Viskositätskonstante                                   ---> 3.78
        %   f_p      -> [mN/g] Zeitabhängige Muskelkraft vom parasympathikus             ---> Optimieren <---
        %   f_s      -> [mN/g] Zeitabhängige Muskelkraft vom sympathikus                 ---> Optimieren <---
        %   P_0      -> [mN/g] Statische Kraft der Irismuskel                            ---> Optimieren <---
        %   tp       -> [s] Verzögerungsparameter parasympathikus                        ---> Optimieren <---
        %   ts       -> [s] Verzögerungsparameter sympathikus                            ---> Optimieren <---
        %   f_p_i_0  -> [mN/g] Irismuskelkradt parasympathikus ohne Reiz                 ---> Optimieren <---
        %   f_s_i_0  -> [mN/g] Irismuskelkradt sympathikus ohne Reiz                     ---> Optimieren <---
        %   x0_1     -> [mm] Lösung der Differentialgleichung Initialpupillendurchmesser ---> D0 Pupil
        %   x0_2     -> [mm] Lösung der Differentialgleichung Initialgeschwindigkeit     ---> 0
        %   Delta_tp -> [] Einschaltzeit des Reizes so wie es aussieht                   ---> 300s (Unsicher)
        % 	Delta_ts -> [] Einschaltzeit des Reizes so wie es aussieht                   ---> 300s (Unsicher)
        %
        % Konstanten definieren
        %   L_0d = [3.95 3.59 3.57];
        %   l_0c = [0.91 0.92 0.93];
        %   K_d = [1.24 1.29 1.24];
        %   K_c = [0.059 0.065 0.047];
        %   D = [3.78 3.78 3.78];
        %   f_p = [12.73 28.38 32.25];
        %   f_s = [9.10 10.65 9.43];
        %   P_0 = [-0.760 -0.736 -1.007];
        %   tp = [0.17 0.15 0.12];
        %   ts = [0.57 0.64 0.75];
        % Delta_tp, Delta_ts entsprechen der Einschaltzeit (Im Paper so definiert)
        %   Delta_tp = 0.1;
        %   Delta_ts = 0.1;
        % -----------------------------------------------------------------------------------------------------
        
        x0 = [x0_1, x0_2];
        
        % Zur nummerischen Lösung der Differentialgleichung
        [T, Sv] = ode45(@ddefunc, t, x0);
        Result = Sv(:,1);
        
        function [yp] = ddefunc(t, y)
            
            % Zeitbedingungen nach dem Paper
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
            
            % Umgestellte Differentialgleichung
            yp = [y(2);...
                (-K_c*(l_0c - y(1))^(2))+(K_d*(L_0d - y(1))^2)-(D*y(2)) - f_p_i_g + f_s_i_g + P_0];
        end
    end