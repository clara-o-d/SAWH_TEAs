function fB = licl_equilibrium_brine_fraction(relativeHumidity, temperatureC)
%LICL_EQUILIBRIUM_BRINE_FRACTION  Invert LiCl isotherm for RH equilibrium.

if nargin < 2
    temperatureC = 25.0;
end

rh = relativeHumidity;
if rh <= 0.0
    fB = 1.0;
    return
end
if rh >= 0.99
    fB = 0.01;
    return
end

lo = 0.01;
hi = 0.75;
for k = 1:60
    mid = 0.5 * (lo + hi);
    aw = licl_water_activity(mid, temperatureC);
    if ~isfinite(aw) || aw < rh
        hi = mid;
    else
        lo = mid;
    end
end
fB = 0.5 * (lo + hi);
end
