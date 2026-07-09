function diaz_marin_figure3()
%DIAZ_MARIN_FIGURE3  Díaz-Marín et al. (2024) Fig. 3 panels B, C, E.
%
%   Self-contained: computes Eq. 5 model curves in MATLAB and overlays
%   digitized reference CSVs. No Python export required.
%
%   Outputs (TIFF, 600 dpi) under diaz-marin-et-al._re-creation/outputs/figure3/:
%     figure3.tif, figure3b.tif, figure3c.tif, figure3e.tif

setup_recreation();
paths = recreation_paths();
refDir = fullfile(paths.diazRef, 'figure3');
outDir = fullfile(paths.diazDir, 'outputs', 'figure3');
if ~isfolder(outDir)
    mkdir(outDir);
end

colSl0 = [0.29, 0.29, 0.29];
colSl1 = [0.12, 0.47, 0.71];
colSl4 = [0.84, 0.15, 0.16];
colSl8 = [0.17, 0.63, 0.17];
colPam = colSl4;
colPva = colSl1;
colXl = [0.58, 0.40, 0.74];
refLabel = 'Díaz-Marín et al. (digitized)';
alpha = 0.7;

fig = figure('Color', 'w', 'Name', 'Díaz-Marín Fig. 3 (B,C,E)');
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_panel_b(nexttile(tl), refDir, colSl0, colSl1, colSl4, colSl8, refLabel);
plot_panel_c(nexttile(tl), refDir, colPam, colPva, refLabel);
plot_panel_e(nexttile(tl), refDir, colXl, refLabel);
sgtitle(fig, { ...
    'Díaz-Marín et al. (2024) Figure 3 — equilibrium uptake isotherms', ...
    'solid = Eq. 5 with LiCl brine isotherm; open circles = digitized paper data'}, ...
    'FontSize', 11);
export_recreation_figure(fig, fullfile(outDir, 'figure3'), alpha, [1, 3]);
pause_for_figure(fig);
if ishandle(fig), close(fig); end

figB = figure('Color', 'w', 'Name', 'Díaz-Marín Fig. 3B');
plot_panel_b(gca, refDir, colSl0, colSl1, colSl4, colSl8, refLabel);
PrintFigure(fullfile(outDir, 'figure3b'), alpha);
pause_for_figure(figB);
if ishandle(figB), close(figB); end

figC = figure('Color', 'w', 'Name', 'Díaz-Marín Fig. 3C');
plot_panel_c(gca, refDir, colPam, colPva, refLabel);
PrintFigure(fullfile(outDir, 'figure3c'), alpha);
pause_for_figure(figC);
if ishandle(figC), close(figC); end

figE = figure('Color', 'w', 'Name', 'Díaz-Marín Fig. 3E');
plot_panel_e(gca, refDir, colXl, refLabel);
PrintFigure(fullfile(outDir, 'figure3e'), alpha);
pause_for_figure(figE);
if ishandle(figE), close(figE); end

fprintf('Saved Díaz-Marín Figure 3 → %s\n', outDir);
end


function plot_panel_b(ax, refDir, colSl0, colSl1, colSl4, colSl8, refLabel)
hold(ax, 'on');
series = {
    0.0, '3b_0.csv', colSl0, 'PAM (0 g/g)'
    1.0, '3b_1.csv', colSl1, 'PAM--LiCl (1 g/g)'
    4.0, '3b_4.csv', colSl4, 'PAM--LiCl (4 g/g)'
    8.0, '3b_8.csv', colSl8, 'PAM--LiCl (8 g/g)'
    };
refShown = false;
for i = 1:size(series, 1)
    sl = series{i, 1};
    csvName = series{i, 2};
    color = series{i, 3};
    label = series{i, 4};
    if sl > 0
        plot_model_curve(ax, sl, color, label);
    end
    if ~refShown
        if overlay_ref(ax, refDir, csvName, color, refLabel)
            refShown = true;
        end
    else
        overlay_ref(ax, refDir, csvName, color, '');
    end
end
style_uptake_axes(ax, 'B  salt content');
end


function plot_panel_c(ax, refDir, colPam, colPva, refLabel)
hold(ax, 'on');
plot_model_curve(ax, 4.0, colPam, 'PAM--LiCl (4 g/g)', '-');
plot_model_curve(ax, 4.0, colPva, 'PVA--LiCl (4 g/g)', '--');
overlay_ref(ax, refDir, '3c.csv', colPam, refLabel);
overlay_ref(ax, refDir, '3c.csv', colPva, '');
style_uptake_axes(ax, 'C  polymer chemistry');
end


function plot_panel_e(ax, refDir, colXl, refLabel)
hold(ax, 'on');
plot_model_curve(ax, 4.0, colXl, 'PAM--LiCl (4 g/g)', '-');
overlay_ref(ax, refDir, '3e.csv', colXl, refLabel);
style_uptake_axes(ax, 'E  crosslinking density');
end


function plot_model_curve(ax, sl, color, label, linestyle)
if nargin < 5
    linestyle = '-';
end
rh = linspace(0, 0.92, 200);
uptake = arrayfun(@(r) diaz_marin_uptake_eq5(r, sl, 25.0), rh);
plot(ax, rh * 100, uptake, 'Color', color, 'LineStyle', linestyle, ...
    'DisplayName', sprintf('%s (model)', label));
end


function style_uptake_axes(ax, panelTitle)
xlabel(ax, 'relative humidity [%]');
ylabel(ax, 'uptake [g/g]');
xlim(ax, [0, 95]);
ylim(ax, [-0.2, 10.0]);
box(ax, 'on');
legend(ax, 'Location', 'northwest', 'Box', 'off', 'FontSize', 10);
title(ax, panelTitle, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
end
