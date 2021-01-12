%% Description:
% Demo code from the article:
% Deep learning based pupil model predicts time and wavelength dependent light responses
% Technical University of Darmstadt, Laboratory of Lighting Technology
% Published in Scientific Reports
% Link: www.nature.com/articles/s41598-020-79908-5
% GitHub Link: https://github.com/BZandi/DL-PupilModel

% Three tools are provided:
% 
% First, the Python code is provided, which was used
% for training the neural networks. This folder also contains
% the training data. Go to the folder: "/Python".

% Secondly, a Matlab GUI is provided, in which the model parameters of the Fan % Yao funktion and masc function
% can be changed. Model parameters and the original median pupil data are stored in the GUI. 
% With this GUI the training data was created.

% Thirdly, the entire combined pupil model with the neural networks is made available
% as an implemented Matlab method. With this method it is possible to reconstruct the temporal pupil diameter
% from the three light conditions 450 nm, 530 nm, 610 nm, 660 nm, 2000 K, 5000 K, 10 000 K at approx. 100 cd/m2.
% Examples of how to call this method can be found here in this main file. The main file is divided into
% 7 different sections (0 to 6), so please do not run the entire Matlab file. Run the section by pressing 
% the "Run section" button (Editor Tab) inside the respective section.

% Since the neural networks
% are implemented in Python, it is necessary that you install Python, otherwise the program will not work.
% Attached is a short manual:

% Step 1: Go to the Functions folder and open the NNCombinedModel.m file.
% Here you have to add your Python Path in line 4. At this point: 
% python_path = '/Users/papillon/opt/anaconda3/envs/ML/bin/python3'; 
% replace this line with your own path. Your Python distribution must have the following 
% packages installed: numpy, pandas, Pytorch, argparse, Pytorch Lightning.

% Step 2: after changing the path go to the first section "0. load data, 
% add all folder to path and set plot preferences" and execute this section.

% Step 3: 6 more sections are then available. There are two different versions
% of the combined pupil model available. One version was trained with the pupil
% data of one subject (intrapersonal) and the second version was trained with the
% pupil data of 20 subjects (interpersonal). For each of these versions three variants
% of neural networks were created. These differ in the input parameters. Variant 1 
% gets the luminance and the CIExy chromaticity points as input. Variant 2 has the receptor
% signals and the melanopsin signal as input. Variant 3 has the luminance, the chromaticity
% coordinates and the melanopsin signal as input. From section 1 to 3 there are variants 1 to 3
% for the intrapersonal data set (1 subject) From section 3 to 6 there are variants 1 to 3 for
% the interpersonal data set (20 subjects). 

% Step 4: If you execute one of these sections (1 - 6), the pupil diameter
% is reconstructed and plotted for all seven light conditions. This plot also contains the measured
% median pupil diameter for comparison. This was used as the target data set in the modeling.

%% 0. Load data, add all folder to path and set plot preferences
addpath("Data");
addpath("Functions");
addpath("Python");

set(groot, 'DefaultLineLineWidth', 1);
set(groot, 'DefaultAxesLineWidth', 1);
set(groot, 'DefaultAxesFontName', 'CMU Serif');
set(groot, 'DefaultAxesFontSize', 14);
set(groot, 'DefaultAxesFontWeight', 'normal');
set(groot, 'DefaultAxesXMinorTick', 'on');
set(groot, 'DefaultAxesXGrid', 'on');
set(groot, 'DefaultAxesYGrid', 'on');
set(groot, 'DefaultAxesGridLineStyle', ':');
set(groot, 'DefaultAxesUnits', 'normalized');
set(groot, 'DefaultAxesOuterPosition',[0, 0, 1, 1]);
set(groot, 'DefaultFigureUnits', 'inches');
set(groot, 'DefaultFigurePaperPositionMode', 'manual');
set(groot, 'DefaultFigurePosition', [0.1, 11, 8.5, 4.5]);
set(groot, 'DefaultFigurePaperUnits', 'inches');
set(groot, 'DefaultFigurePaperPosition', [0.1, 11, 8.5, 4.5]);

%% 1. Predict & Plot PLR from model with neural network - Single subject (Variant 1): L, CIExy-2°
clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
Luminance = [99.73; 100.12; 100.16; 99.97; 100.17; 100.10; 99.83];
CIE_x = [0.15811; 0.18661; 0.67903; 0.71701; 0.53305; 0.34538; 0.28549];
CIE_y = [0.02006; 0.73928; 0.32039; 0.27995; 0.42288; 0.34976; 0.2769];

hparam.Condition = 'Single';
hparam.Variant = 1;

% Light conditino #1
Index = 1;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (530 nm)' 'L, CIExy 2°'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (610 nm)' 'L, CIExy 2°'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (660 nm)' 'L, CIExy 2°'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (2000 K)' 'L, CIExy 2°'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (5000 K)' 'L, CIExy 2°'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (10 000 K)' 'L, CIExy 2°'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

%% 2. Predict & Plot PLR from model with neural network - Single subject (Variant 2): Lcone, Mcone, Scone, Mel
clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
S_Signal = [3.24582; 0.00396; 9.00e-05; 0.00084; 0.00637; 0.06364; 0.11978];
M_Signal = [0.48338; 0.16724; 0.04708; 0.02363; 0.10258; 0.13957; 0.15241];
L_Signal = [0.2913; 0.14091; 0.19684; 0.20227; 0.17090; 0.16452; 0.16520];
Melanopsin_Signal = [1.81725; 0.10547; 0.00075; 0.00114; 0.04561; 0.12247; 0.1668];

hparam.Condition = 'Single';
hparam.Variant = 2;

% Light conditino #1
Index = 1;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L-, M-, S-cone, Mel'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (530 nm)' 'L-, M-, S-cone, Mel'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (610 nm)' 'L-, M-, S-cone, Mel'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (660 nm)' 'L-, M-, S-cone, Mel'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (2000 K)' 'L-, M-, S-cone, Mel'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (5000 K)' 'L-, M-, S-cone, Mel'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (10 000 K)' 'L-, M-, S-cone, Mel'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

%% 3. Predict & Plot PLR from model with neural network - Single subject (Variant 3): L, CIExy-2°, Mel
clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
Luminance = [99.73; 100.12; 100.16; 99.97; 100.17; 100.10; 99.83];
CIE_x = [0.15811; 0.18661; 0.67903; 0.71701; 0.53305; 0.34538; 0.28549];
CIE_y = [0.02006; 0.73928; 0.32039; 0.27995; 0.42288; 0.34976; 0.2769];
Melanopsin_Signal = [1.81725; 0.10547; 0.00075; 0.00114; 0.04561; 0.12247; 0.1668];

hparam.Condition = 'Single';
hparam.Variant = 3;

% Light conditino #1
Index = 1;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_OneSubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intrasubject (450 nm)' 'L, CIExy 2°, Mel'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

%% 4. Predict & Plot PLR from model with neural network - Multi subjects (Variant 1): L, CIExy-2°

clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
Luminance = [99.73; 100.12; 100.16; 99.97; 100.17; 100.10; 99.83];
CIE_x = [0.15811; 0.18661; 0.67903; 0.71701; 0.53305; 0.34538; 0.28549];
CIE_y = [0.02006; 0.73928; 0.32039; 0.27995; 0.42288; 0.34976; 0.2769];

hparam.Condition = 'Multi';
hparam.Variant = 1;

% Light conditino #1
Index = 1;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (530 nm)' 'L, CIExy 2°'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (610 nm)' 'L, CIExy 2°'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (660 nm)' 'L, CIExy 2°'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (2000 K)' 'L, CIExy 2°'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (5000 K)' 'L, CIExy 2°'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (10 000 K)' 'L, CIExy 2°'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

%% 5. Predict & Plot PLR from model with neural network - Multi subjects (Variant 2): Receptor signals, Mel
clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
S_Signal = [3.24582; 0.00396; 9.00e-05; 0.00084; 0.00637; 0.06364; 0.11978];
M_Signal = [0.48338; 0.16724; 0.04708; 0.02363; 0.10258; 0.13957; 0.15241];
L_Signal = [0.2913; 0.14091; 0.19684; 0.20227; 0.17090; 0.16452; 0.16520];
Melanopsin_Signal = [1.81725; 0.10547; 0.00075; 0.00114; 0.04561; 0.12247; 0.1668];

hparam.Condition = 'Multi';
hparam.Variant = 2;

% Light conditino #1
Index = 1;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [L_Signal(Index), M_Signal(Index), S_Signal(Index), Melanopsin_Signal(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L-, M-, S-cone, Mel'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (530 nm)' 'L-, M-, S-cone, Mel'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (610 nm)' 'L-, M-, S-cone, Mel'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (660 nm)' 'L-, M-, S-cone, Mel'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (2000 K)' 'L-, M-, S-cone, Mel'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (5000 K)' 'L-, M-, S-cone, Mel'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (10 000 K)' 'L-, M-, S-cone, Mel'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

%% 6. Predict & Plot PLR from model with neural network - Multi subjects (Variant 3): L, CIExy-2°, Mel
clc; clear; load("Data");

% Light conditions : '450nm'; '530nm'; '610nm'; '660nm'; '2000K'; '5000K'; '10000K'
Luminance = [99.73; 100.12; 100.16; 99.97; 100.17; 100.10; 99.83];
CIE_x = [0.15811; 0.18661; 0.67903; 0.71701; 0.53305; 0.34538; 0.28549];
CIE_y = [0.02006; 0.73928; 0.32039; 0.27995; 0.42288; 0.34976; 0.2769];
Melanopsin_Signal = [1.81725; 0.10547; 0.00075; 0.00114; 0.04561; 0.12247; 0.1668];

hparam.Condition = 'Multi';
hparam.Variant = 3;

% Light conditino #1
Index = 1;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted1] = NNCombinedModel(hparam);

% Light conditino #2
Index = 2;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted2] = NNCombinedModel(hparam);

% Light conditino #3
Index = 3;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted3] = NNCombinedModel(hparam);

% Light conditino #4
Index = 4;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted4] = NNCombinedModel(hparam);

% Light conditino #5
Index = 5;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted5] = NNCombinedModel(hparam);

% Light conditino #6
Index = 6;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted6] = NNCombinedModel(hparam);

% Light conditino #7
Index = 7;
hparam.Stimuli = [Luminance(Index), CIE_x(Index), CIE_y(Index), Melanopsin_Signal(Index)];
[out, Predicted7] = NNCombinedModel(hparam);

% Plot predicted PLR with measured PLR
figure;
t = tiledlayout(3,3, 'Padding',"none", "TileSpacing","normal");
set(gcf, 'Position', [0.1, 11, 14, 8]);
t.Padding = "none";
t.TileSpacing = "compact";
LinewidthModel = 1;

ax1 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum450nm.Median, 'k'); hold on;
plot(Predicted1.Time, Predicted1.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})
legend({'Measured PD', 'Predicted PD'},'Location','northwest','NumColumns', 2)

ax2 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum530nm.Median, 'k'); hold on;
plot(Predicted2.Time, Predicted2.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

ax3 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum610nm.Median, 'k'); hold on;
plot(Predicted3.Time, Predicted3.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

ax4 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.Spektrum660nm.Median, 'k'); hold on;
plot(Predicted4.Time, Predicted4.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

ax5 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT2000K.Median, 'k'); hold on;
plot(Predicted5.Time, Predicted5.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

ax6 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT5000K.Median, 'k'); hold on;
plot(Predicted6.Time, Predicted6.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

ax7 = nexttile;
plot(linspace(0, 300, 30001)', Pupillendaten_Wide_ManySubject.CCT10000K.Median, 'k'); hold on;
plot(Predicted7.Time, Predicted7.CombinedModel, 'b', 'LineWidth', LinewidthModel);
xlabel('Time in seconds'); ylabel('PD in mm'); title({'Intersubject (450 nm)' 'L, CIExy 2°, Mel'})

set(ax1, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax2, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax3, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax4, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax5, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax6, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');
set(ax7, 'YLim', [1, 5], 'XLim', [0, 300]); grid(ax1, 'off');

