function uptake = diaz_marin_uptake_eq5(relativeHumidity, saltToPolymerRatio, temperatureC)
%DIAZ_MARIN_UPTAKE_EQ5  Eq. 5 gravimetric uptake U [g/g] for PAM--LiCl.

if nargin < 3
    temperatureC = 25.0;
end

rh = relativeHumidity;
sl = saltToPolymerRatio;
if sl <= 0.0 || rh <= 0.0
    uptake = 0.0;
    return
end

mwW = 18.01528;   % g/mol
mwS = 42.394;     % LiCl g/mol
ions = 2;

fB = licl_equilibrium_brine_fraction(rh, temperatureC);
if ~isfinite(fB) || fB <= 0.0
    uptake = nan;
    return
end

massWater = 1.0 - fB;
massSalt = fB;
nW = massWater / mwW;
nS = massSalt / mwS;
xW = nW / (nW + ions * nS);
if xW >= 1.0
    uptake = nan;
    return
end

uSalt = (xW * ions) / (1.0 - xW) * (mwW / mwS);
polymerFactor = sl / (1.0 + sl);
uptake = polymerFactor * uSalt;
end
