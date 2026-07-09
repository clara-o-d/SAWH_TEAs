function run_all_recreation_figures()
%RUN_ALL_RECREATION_FIGURES  Plot all paper recreation figures in MATLAB.
%
%   Workflow:
%     1. cd to solar_lumped in a terminal
%     2. python scripts/export_recreation_matlab_data.py   (simulation-heavy figures)
%     3. In MATLAB:
%          cd('<repo>/solar_lumped/matlab/recreation')
%          run_all_recreation_figures
%
%   Díaz-Marín Figure 3 runs without the Python export step (Eq. 5 in MATLAB).

setup_recreation();
fprintf('=== Díaz-Marín Figure 3 (self-contained) ===\n');
diaz_marin_figure3();

if isfolder(fullfile(recreation_paths().diazMatlab, 'figure5'))
    fprintf('=== Díaz-Marín Figure 5 ===\n');
    diaz_marin_figure5();
else
    fprintf('Skipping Díaz-Marín Figure 5 (run export_recreation_matlab_data.py --figures diaz5)\n');
end

if isfolder(fullfile(recreation_paths().wilsonMatlab, 'figure3'))
    fprintf('=== Wilson Figure 3 ===\n');
    wilson_figure3();
else
    fprintf('Skipping Wilson Figure 3 (run export ... --figures wilson3)\n');
end

if isfolder(fullfile(recreation_paths().wilsonMatlab, 'figure4'))
    fprintf('=== Wilson Figure 4 ===\n');
    wilson_figure4();
else
    fprintf('Skipping Wilson Figure 4 (run export ... --figures wilson4)\n');
end

if isfolder(fullfile(recreation_paths().wilsonMatlab, 'figure2'))
    fprintf('=== Wilson Figure 2 ===\n');
    wilson_figure2();
else
    fprintf('Skipping Wilson Figure 2 (run figure2_generate.py + export ... --figures wilson2)\n');
end

fprintf('Done.\n');
end
