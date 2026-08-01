function wilson_figure4()
%WILSON_FIGURE4  Wilson et al. (2025) Fig. 4 panels C and D (Atacama).
%
%   Requires model CSV from Python:
%     python scripts/export_recreation_matlab_data.py --figures wilson4

setup_recreation();
paths = recreation_paths();
modelDir = fullfile(paths.wilsonMatlab, 'figure4');
refDir = fullfile(paths.wilsonRef, 'figure4');
assert(isfolder(modelDir), ...
    'Missing model data. Run: python scripts/export_recreation_matlab_data.py --figures wilson4');

outDir = fullfile(paths.wilsonDir, 'outputs', 'figure4');
if ~isfolder(outDir)
    mkdir(outDir);
end

model = readmatrix(fullfile(modelDir, 'model.csv'));
metaLine = strtrim(fileread(fullfile(modelDir, 'meta.txt')));
etaPct = sscanf(metaLine, 'eta=%f');
refLabel = 'Wilson et al. (digitized)';
measuredYield = 0.62;
alpha = 0.7;

colAbs = [0.75, 0.22, 0.17];
colGlass = [0.90, 0.49, 0.13];
colCond = [0.16, 0.50, 0.73];
colAmb = [0.50, 0.55, 0.55];
colWater = [0.10, 0.74, 0.61];

timeH = model(:, 1);
fig = figure('Color', 'w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

axC = nexttile(tl);
hold(axC, 'on');
plot(axC, timeH, model(:, 2), 'Color', colAbs, 'DisplayName', 'absorber (model)');
plot(axC, timeH, model(:, 3), '--', 'Color', colGlass, 'DisplayName', 'glass (model)');
plot(axC, timeH, model(:, 4), 'Color', colCond, 'DisplayName', 'condenser (model)');
plot(axC, timeH, model(:, 5), ':', 'Color', colAmb, 'DisplayName', 'T_{amb} (measured)');
overlay_ref(axC, refDir, 'Atacama_absorber copy.csv', colAbs, refLabel);
overlay_ref(axC, refDir, 'Atacama_glass copy.csv', colGlass, '');
overlay_ref(axC, refDir, 'Atacama_condenser copy.csv', colCond, '');
ylabel(axC, 'Temperature (°C)');
xlabel(axC, 'Time from 8 am (hr)');
xlim(axC, [0, timeH(end)]);
ylim(axC, [0, inf]);
box(axC, 'on');
legend(axC, 'Location', 'northwest', 'Box', 'off', 'FontSize', 9);
title(axC, 'C', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

axD = nexttile(tl);
hold(axD, 'on');
plot(axD, timeH, model(:, 6), 'Color', colWater, 'LineWidth', 2.0, ...
    'DisplayName', 'water output (model)');
plot(axD, timeH(end), measuredYield, '*', 'Color', [0.91, 0.30, 0.24], ...
    'MarkerSize', 11, 'DisplayName', sprintf('Measured (%.2f L/m^2)', measuredYield));
ylabel(axD, 'Cumulative water output (L/m^2)');
xlabel(axD, 'Time from 8 am (hr)');
xlim(axD, [0, timeH(end)]);
ylim(axD, [0, inf]);
box(axD, 'on');
legend(axD, 'Location', 'northwest', 'Box', 'off', 'FontSize', 9);
title(axD, 'D', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

yieldVal = model(end, 6);
sgtitle(fig, sprintf([ ...
    'Wilson et al. (2025) Figure 4 — Atacama Desert field test (May 2024)\n' ...
    'Model yield = %.3f L/m^2 (measured 0.62 L/m^2),  \\eta_{th} = %.1f%%;  open circles = digitized data'], ...
    yieldVal, etaPct(1)), 'FontSize', 10);

export_recreation_figure(fig, fullfile(outDir, 'figure4'), alpha, [1, 2]);
pause_for_figure(fig);
if ishandle(fig), close(fig); end
fprintf('Saved Wilson Figure 4 → %s\n', outDir);
end

