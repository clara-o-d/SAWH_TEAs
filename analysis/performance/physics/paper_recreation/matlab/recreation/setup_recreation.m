function setup_recreation()
%SETUP_RECREATION  Add MATLAB helpers and set slide plot defaults.

paths = recreation_paths();
addpath(paths.matlabDir);
addpath(fileparts(mfilename('fullpath')));
PlotDefaults_Slides();
end
