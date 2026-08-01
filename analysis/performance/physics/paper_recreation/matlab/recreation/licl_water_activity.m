function aw = licl_water_activity(brineSaltFraction, temperatureC)
%LICL_WATER_ACTIVITY  Conde (2004) LiCl brine water activity (Python port).

f = brineSaltFraction;
if ~(isfinite(f) && f >= 0.0 && f < 1.0)
    aw = nan;
    return
end

tCorr = min(temperatureC, 150.0);
tr = (tCorr + 273.15) / 647.0;

p0 = 0.28; p1 = 4.3; p2 = 0.60;
p3 = 0.21; p4 = 5.10; p5 = 0.49;
p6 = 0.362; p7 = -4.75; p8 = -0.40; p9 = 0.03;

concentrationTerm = 1.0 ...
    - (1.0 + (f / p6) ^ p7) ^ p8 ...
    - p9 * exp(-((f - 0.1) ^ 2) / 0.005);

temperatureTerm = 2.0 ...
    - (1.0 + (f / p0) ^ p1) ^ p2 ...
    + ((1.0 + (f / p3) ^ p4) ^ p5 - 1.0) * tr;

aw = max(0.0, min(1.0, concentrationTerm * temperatureTerm));
end
