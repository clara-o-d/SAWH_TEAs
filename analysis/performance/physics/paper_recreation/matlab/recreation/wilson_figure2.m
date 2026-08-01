function wilson_figure2()
%WILSON_FIGURE2  Wilson et al. (2025) Fig. 2 panels B–F.
%
%   Requires sweep model CSVs from Python:
%     python wilson-et-al._re-creation/scripts/figure2_generate.py
%     python scripts/export_recreation_matlab_data.py --figures wilson2

setup_recreation();
paths = recreation_paths();
modelDir = fullfile(paths.wilsonMatlab, 'figure2');
refDir = fullfile(paths.wilsonRef, 'figure2');
assert(isfolder(modelDir), ...
    'Missing model data. Run figure2_generate.py then export_recreation_matlab_data.py --figures wilson2');

outDir = fullfile(paths.wilsonDir, 'outputs', 'figure2');
if ~isfolder(outDir)
    mkdir(outDir);
end

refLabel = 'Wilson et al. (digitized)';
fig = figure('Color', 'w');
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(tl);
axis off;
text(0.5, 0.5, {'Device'; 'schematic'; '(see Fig. 2A'; 'in paper)'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0.5 0.5 0.5]);
title('A', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

plot_panel_b(nexttile(tl), modelDir, refDir, refLabel);
plot_panel_c(nexttile(tl), modelDir, refDir, refLabel);
plot_panel_d(nexttile(tl), modelDir, refDir, refLabel);
plot_panel_e(nexttile(tl), modelDir, refDir, refLabel);
plot_panel_f(nexttile(tl), modelDir, refDir, refLabel);

sgtitle(fig, { ...
    'Wilson et al. (2025) Figure 2 — Thermofluidic optimisation of the hydrogel SAWH device', ...
    'dashed = solar\_lumped model; circles = digitized paper data; bands = h_{amb}=10±2.5 W/m^2K'}, ...
    'Interpreter', 'tex', 'FontSize', 11);

export_recreation_figure(fig, fullfile(outDir, 'figure2'), 0.7, [2, 3]);
pause_for_figure(fig);
if ishandle(fig), close(fig); end
fprintf('Saved Wilson Figure 2 → %s\n', outDir);
end


function plot_panel_b(ax, modelDir, refDir, refLabel)
hold(ax, 'on');
epsVals = [0.2, 0.5, 0.8, 0.95, 0.99];
cols = lines(numel(epsVals));
refShown = false;
for i = 1:numel(epsVals)
    eps = epsVals(i);
    data = readmatrix(fullfile(modelDir, sprintf('2b_eps_%.2f.csv', eps)));
    plot(ax, data(:,1), data(:,2), '--', 'Color', cols(i,:), ...
        'DisplayName', sprintf('\\epsilon_{abs}=%.2f (model)', eps));
    if ~refShown
        overlay_ref(ax, refDir, sprintf('2b_%.2f.csv', eps), cols(i,:), refLabel);
        refShown = true;
    else
        overlay_ref(ax, refDir, sprintf('2b_%.2f.csv', eps), cols(i,:), '');
    end
end
xlabel(ax, 'transmission through glass, \tau_{glass}');
ylabel(ax, 'device productivity [L/m^2/day]');
xlim(ax, [0.2, 1.0]);
style_panel(ax, 'B');
end


function plot_panel_c(ax, modelDir, refDir, refLabel)
hold(ax, 'on');
arVals = [1, 2, 5, 7];
refShown = false;
for i = 1:numel(arVals)
    ar = arVals(i);
    dataG = readmatrix(fullfile(modelDir, sprintf('2c_ar_%d_glass.csv', ar)));
    plot(ax, dataG(:,1), dataG(:,2), '--', 'DisplayName', sprintf('A_r=%d (model)', ar));
    if ~refShown
        overlay_ref(ax, refDir, sprintf('2c_%d_glass.csv', ar), [0 0 0], refLabel);
        refShown = true;
    else
        overlay_ref(ax, refDir, sprintf('2c_%d_glass.csv', ar), [0 0 0], '');
    end
    dataN = readmatrix(fullfile(modelDir, sprintf('2c_ar_%d_noglass.csv', ar)));
    plot(ax, dataN(:,1), dataN(:,2), ':', 'Color', [0.6 0.6 0.6]);
    overlay_ref(ax, refDir, sprintf('2c_%d_no-glass.csv', ar), [0.6 0.6 0.6], '');
end
xlabel(ax, 'ambient heat transfer coefficient, h_{amb} [W/m^2K]');
ylabel(ax, 'device productivity [L/m^2/day]');
xlim(ax, [1, 10]);
style_panel(ax, 'C');
end


function plot_panel_d(ax, modelDir, refDir, refLabel)
hold(ax, 'on');
tVals = [280, 290, 300, 310];
cols = lines(numel(tVals));
refShown = false;
for i = 1:numel(tVals)
    Tk = tVals(i);
    data = readmatrix(fullfile(modelDir, sprintf('2d_T_%d.csv', Tk)));
    plot(ax, data(:,1), data(:,2), '--', 'Color', cols(i,:), ...
        'DisplayName', sprintf('T_{amb}=%d K (model)', Tk));
    if ~refShown
        overlay_ref(ax, refDir, sprintf('2d_%d.csv', Tk), cols(i,:), refLabel);
        refShown = true;
    else
        overlay_ref(ax, refDir, sprintf('2d_%d.csv', Tk), cols(i,:), '');
    end
end
xlabel(ax, 'ambient humidity RH [ ]');
ylabel(ax, 'device productivity [L/m^2/day]');
xlim(ax, [0.2, 0.9]);
style_panel(ax, 'D');
end


function plot_panel_e(ax, modelDir, refDir, refLabel)
hold(ax, 'on');
lgVals = [18, 20, 40, 60];
cols = lines(numel(lgVals));
refShown = false;
for i = 1:numel(lgVals)
    lg = lgVals(i);
    data = readmatrix(fullfile(modelDir, sprintf('2e_Lg_%d.csv', lg)));
    plot(ax, data(:,1), data(:,2), '--', 'Color', cols(i,:), ...
        'DisplayName', sprintf('L_g=%d mm (model)', lg));
    if ~refShown
        overlay_ref(ax, refDir, sprintf('2e_%d.csv', lg), cols(i,:), refLabel);
        refShown = true;
    else
        overlay_ref(ax, refDir, sprintf('2e_%d.csv', lg), cols(i,:), '');
    end
end
xlabel(ax, 'thickness of gel, H_0 [mm]');
ylabel(ax, 'device productivity [L/m^2/day]');
xlim(ax, [0, 8]);
style_panel(ax, 'E');
end


function plot_panel_f(ax, modelDir, refDir, refLabel)
hold(ax, 'on');
ax2 = yyaxis(ax, 'right');
h0Vals = [2, 4, 8];
refShown = false;
yyaxis(ax, 'left');
for i = 1:numel(h0Vals)
    h0 = h0Vals(i);
    prod = readmatrix(fullfile(modelDir, sprintf('2f_prod_%d.csv', h0)));
    plot(ax, prod(:,1), prod(:,2), '--o', 'DisplayName', sprintf('H_0=%d mm yield (model)', h0));
    if ~refShown
        overlay_ref(ax, refDir, sprintf('2f_prod_%d.csv', h0), [0 0 0], refLabel);
        refShown = true;
    else
        overlay_ref(ax, refDir, sprintf('2f_prod_%d.csv', h0), [0 0 0], '');
    end
end
ylabel(ax, 'device productivity [L/m^2/day]');
yyaxis(ax, 'right');
for i = 1:numel(h0Vals)
    h0 = h0Vals(i);
    eff = readmatrix(fullfile(modelDir, sprintf('2f_eff_%d.csv', h0)));
    plot(ax, eff(:,1), eff(:,2), '--x');
    overlay_ref(ax, refDir, sprintf('2f_eff_%d.csv', h0), [0.5 0.5 0.5], '');
end
ylabel(ax, 'thermal efficiency [%]');
xlabel(ax, 'incident solar flux, Q_{solar} [kW/m^2]');
xlim(ax, [0.5, 1.5]);
style_panel(ax, 'F');
end


function style_panel(ax, panelLetter)
box(ax, 'on');
legend(ax, 'Location', 'best', 'Box', 'off', 'FontSize', 8);
title(ax, panelLetter, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
end

