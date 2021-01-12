%% Description:
% Demo code from the article:
% Deep learning based pupil model predicts time and wavelength dependent light responses
% Technical University of Darmstadt, Laboratory of Lighting Technology
% Published in Scientific Reports
% Link: www.nature.com/articles/s41598-020-79908-5
% GitHub Link: https://github.com/BZandi/DL-PupilModel

function result = Masc_function(x, q, r)

    %Function_1 = @(x) ((0.5.*(-q.*(x-r)))/ (nthroot( (q.*(x-r).^(2)+1),2))) + 0.5;
    %Function_2 = @(x) ((0.5.*(q.*(x-r)))/ (nthroot( (q.*(x-r).^(2)+1),2))) + 0.5;    
    %g_1 = Function_1(x);
    %g_2 = Function_2(x);
    
    Function_tanh_2 = @(x) 0.5 + 0.5 * tanh((x-q)/r);
    Function_tanh_1 = @(x) 1-(0.5 + 0.5 * tanh((x-q)/r));
    
    g_1 = Function_tanh_1(x);
    g_2 = Function_tanh_2(x);
    
    result(:,1) = g_1;
    result(:,2) = g_2;
end