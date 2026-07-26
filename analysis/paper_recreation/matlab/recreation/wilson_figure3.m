function wilson_figure3()
%WILSON_FIGURE3  Wilson et al. (2025) Fig. 3 panels B and C (Cambridge).
%
%   Requires model CSVs from Python:
%     python scripts/export_recreation_matlab_data.py --figures wilson3

setup_recreation();
paths = recreation_paths();
modelDir = fullfile(paths.wilsonMatlab, 'figure3');
refDir = fullfile(paths.wilsonRef, 'figure3');
assert(isfolder(modelDir), ...
    'Missing model data. Run: python scripts/export_recreation_matlab_data.py --figures wilson3');

outDir = fullfile(paths.wilsonDir, 'outputs', 'figure3');
if ~isfolder(outDir)
    mkdir(outDir);
end

colAbs = [0.55, 0.13, 0.00];
colGlass = [0.69, 0.38, 0.00];
colCond = [0.00, 0.44, 0.56];
colAmb = [0.13, 0.25, 0.69];
colWater = [0.10, 0.42, 0.36];
refLabel = 'Wilson et al. (digitized)';
alpha = 0.7;
desHours = 10.0;

mid = readmatrix(fullfile(modelDir, 'h_amb_10.0.csv'));
lo = readmatrix(fullfile(modelDir, 'h_amb_7.5.csv'));
hi = readmatrix(fullfile(modelDir, 'h_amb_12.5.csv'));
weather = readmatrix(fullfile(modelDir, 'weather.csv'));

t = mid(:, 1);
tGrid = weather(:, 1);
tempGrid = weather(:, 3);

fig = figure('Color', 'w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

axB = nexttile(tl);
plot_temp_panel(axB, t, mid, lo, hi, tGrid, tempGrid, colAbs, colGlass, colCond, colAmb, refDir, refLabel, 'B');
xlim(axB, [0, desHours]);
ylim(axB, [10, 80]);

axC = nexttile(tl);
hold(axC, 'on');
plot(axC, t, mid(:, 6), 'Color', colWater, 'DisplayName', 'water output (model)');
aC = 0.078; % Wilson Table S3 A_c [m²] — digitized data is device-total mL
[xRef, yRef] = load_ref_csv(refDir, 'Cambridge_water_output_ml.csv');
if ~isempty(xRef)
    scatter(axC, xRef, yRef / aC, 36, 'o', ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', colWater, ...
        'LineWidth', 1.5, 'DisplayName', refLabel);
end
xlabel(axC, 'time [hr]');
ylabel(axC, 'cumulative water output [mL/m^2]');
xlim(axC, [0, desHours]);
ylim(axC, [0, inf]);
box(axC, 'on');
legend(axC, 'Location', 'northwest', 'Box', 'off', 'FontSize', 10);
title(axC, 'C', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

sgtitle(fig, { ...
    'Wilson et al. (2025) Figure 3 — Cambridge field test', ...
    'model lines; open circles = digitized data; band = h_{amb}=10±2.5 W/m^2K'}, ...
    'FontSize', 11);
export_recreation_figure(fig, fullfile(outDir, 'figure3'), alpha, [1, 2]);
pause_for_figure(fig);
if ishandle(fig), close(fig); end

figB = figure('Color', 'w', 'Name', 'Wilson Fig. 3B');
plot_temp_panel(gca, t, mid, lo, hi, tGrid, tempGrid, colAbs, colGlass, colCond, colAmb, refDir, refLabel, '');
xlim(gca, [0, desHours]);
ylim(gca, [10, 80]);
PrintFigure(fullfile(outDir, 'figure3b'), alpha);
pause_for_figure(figB);
if ishandle(figB), close(figB); end

fprintf('Saved Wilson Figure 3 → %s\n', outDir);
end


function plot_temp_panel(ax, t, mid, lo, hi, tGrid, tempGrid, colAbs, colGlass, colCond, colAmb, refDir, refLabel, panelTitle)
hold(ax, 'on');
fill_band(ax, t, min(lo(:,2), min(mid(:,2), hi(:,2))), max(lo(:,2), max(mid(:,2), hi(:,2))), colAbs);
plot(ax, t, mid(:,2), 'Color', colAbs, 'DisplayName', 'absorber (model)');
fill_band(ax, t, min(lo(:,3), min(mid(:,3), hi(:,3))), max(lo(:,3), max(mid(:,3), hi(:,3))), colGlass);
plot(ax, t, mid(:,3), 'Color', colGlass, 'DisplayName', 'glass (model)');
fill_band(ax, t, min(lo(:,4), min(mid(:,4), hi(:,4))), max(lo(:,4), max(mid(:,4), hi(:,4))), colCond);
plot(ax, t, mid(:,4), 'Color', colCond, 'DisplayName', 'condenser (model)');
plot(ax, tGrid, tempGrid, '--', 'Color', colAmb, 'DisplayName', 'ambient (measured)');
overlay_ref(ax, refDir, 'Cambridge_absorber.csv', colAbs, refLabel);
overlay_ref(ax, refDir, 'Cambridge_glass.csv', colGlass, '');
overlay_ref(ax, refDir, 'Cambridge_condenser.csv', colCond, '');
xlabel(ax, 'time [hr]');
ylabel(ax, 'temperature [°C]');
box(ax, 'on');
legend(ax, 'Location', 'northeast', 'Box', 'off', 'FontSize', 10);
if ~isempty(panelTitle)
    title(ax, panelTitle, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
end
end


function fill_band(ax, x, yLo, yHi, color)
fill(ax, [x; flipud(x)], [yLo; flipud(yHi)], color, ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

