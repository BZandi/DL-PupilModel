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
            case 'Many'
                Offset_Correction = 0.41834;
        end
        pupilDiameterWatson = ResultFunction_1 + ResultB - Offset_Correction;
    end
end