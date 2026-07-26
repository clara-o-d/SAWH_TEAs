function diaz_marin_figure5()
%DIAZ_MARIN_FIGURE5  Díaz-Marín et al. (2024) Fig. 5 panels C, D, E, I.
%
%   Requires model kinetics CSVs exported from Python first:
%     cd solar_lumped
%     python scripts/export_recreation_matlab_data.py --figures diaz5
%
%   Model files: diaz-marin-et-al._re-creation/outputs/matlab/figure5/<panel>_<rh>.csv
%   Reference:   diaz-marin-et-al._re-creation/reference/figure5/

setup_recreation();
paths = recreation_paths();
refDir = fullfile(paths.diazRef, 'figure5');
modelDir = fullfile(paths.diazMatlab, 'figure5');
assert(isfolder(modelDir), ...
    'Missing model data. Run: python scripts/export_recreation_matlab_data.py --figures diaz5');

outDir = fullfile(paths.diazDir, 'outputs', 'figure5');
if ~isfolder(outDir)
    mkdir(outDir);
end

panels = {
    '5c', 'C  PAM--LiCl 4 g/g'
    '5d', 'D  PAM--LiCl 2 g/g'
    '5e', 'E  PVA--LiCl 4 g/g'
    '5i', 'I  PAM--LiCl 4 g/g (~3.2 mm)'
    };
rhList = [30, 50, 70];
rhColors = {
    30, [0.12, 0.47, 0.71]
    50, [0.79, 0.64, 0.15]
    70, [0.17, 0.63, 0.17]
    };
refLabel = 'Díaz-Marín et al. (digitized)';
alpha = 0.7;

fig = figure('Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:size(panels, 1)
  plot_kinetics_panel(nexttile(tl), panels{i, 1}, panels{i, 2}, ...
      modelDir, refDir, rhList, rhColors, refLabel);
end
sgtitle(fig, { ...
    'Díaz-Marín et al. (2024) Figure 5 — absorption–desorption kinetics', ...
    'black = hydrogel Eq. 5 + 8; circles = digitized data'}, 'FontSize', 11);
export_recreation_figure(fig, fullfile(outDir, 'figure5'), alpha, [2, 2]);
pause_for_figure(fig);
if ishandle(fig), close(fig); end

for i = 1:size(panels, 1)
    figP = figure('Color', 'w', 'Name', ['Díaz-Marín Fig. ' panels{i, 1}]);
    plot_kinetics_panel(gca, panels{i, 1}, panels{i, 2}, ...
        modelDir, refDir, rhList, rhColors, refLabel);
    PrintFigure(fullfile(outDir, ['figure' panels{i, 1}]), alpha);
    pause_for_figure(figP);
    if ishandle(figP), close(figP); end
end

fprintf('Saved Díaz-Marín Figure 5 → %s\n', outDir);
end


function plot_kinetics_panel(ax, panelKey, panelTitle, modelDir, refDir, rhList, rhColors, refLabel)
modelLineWidth = 2.5;
refMarkerLineWidth = 2.0;
hold(ax, 'on');
tMax = 0;
refShown = false;
for r = 1:numel(rhList)
    rh = rhList(r);
    color = rhColors{r, 2};
    modelFile = fullfile(modelDir, sprintf('%s_%d.csv', panelKey, rh));
    assert(isfile(modelFile), 'Missing model file: %s', modelFile);
    data = readmatrix(modelFile);
    tMin = data(:, 1);
    uptake = data(:, 2);
    tMax = max(tMax, tMin(end));
    plot(ax, tMin, uptake, 'k-', 'LineWidth', modelLineWidth, 'HandleVisibility', 'off');
    csvName = sprintf('%s_%d.csv', panelKey, rh);
    if ~refShown
        overlay_ref(ax, refDir, csvName, color, refLabel, refMarkerLineWidth);
        refShown = true;
    else
        overlay_ref(ax, refDir, csvName, color, '', refMarkerLineWidth);
    end
end

hRh = gobjects(numel(rhList), 1);
for r = 1:numel(rhList)
    hRh(r) = plot(ax, nan, nan, 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', rhColors{r, 2}, ...
        'LineWidth', refMarkerLineWidth, 'DisplayName', sprintf('20–%d–20 %% RH', rhList(r)));
end
hModel = plot(ax, nan, nan, 'k-', 'LineWidth', modelLineWidth, 'DisplayName', 'model (Eq. 5 + 8)');
legend(ax, [hRh; hModel], 'Location', 'northeast', 'Box', 'off', 'FontSize', 9);

xlabel(ax, 'time [min]');
ylabel(ax, 'uptake [g/g]');
xlim(ax, [0, tMax]);
ylim(ax, [-0.05, 1.05]);
box(ax, 'on');
title(ax, panelTitle, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
end

