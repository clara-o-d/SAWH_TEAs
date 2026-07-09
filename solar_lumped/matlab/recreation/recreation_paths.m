function p = recreation_paths()
%RECREATION_PATHS  Resolve repo folders for recreation figure scripts.
%
%   p = recreation_paths();
%
%   Run setup_recreation.m once per MATLAB session (or let each figure script
%   call it automatically).

scriptDir = fileparts(mfilename('fullpath'));
p.matlabDir = fileparts(scriptDir);
p.solarRoot = fileparts(p.matlabDir);
p.repoRoot = fileparts(p.solarRoot);
p.wilsonDir = fullfile(p.solarRoot, 'wilson-et-al._re-creation');
p.diazDir = fullfile(p.solarRoot, 'diaz-marin-et-al._re-creation');
p.wilsonRef = fullfile(p.wilsonDir, 'reference');
p.diazRef = fullfile(p.diazDir, 'reference');
p.wilsonMatlab = fullfile(p.wilsonDir, 'outputs', 'matlab');
p.diazMatlab = fullfile(p.diazDir, 'outputs', 'matlab');
end
