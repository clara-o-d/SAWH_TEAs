function plotted = overlay_ref(ax, refDir, filename, color, labelText, markerLineWidth)
%OVERLAY_REF  Open-circle markers for digitized paper data.

if nargin < 5
    labelText = '';
end
if nargin < 6
    markerLineWidth = 1.5;
end

[x, y] = load_ref_csv(refDir, filename);
plotted = ~isempty(x);
if ~plotted
    return
end

hold(ax, 'on');
if isempty(labelText)
    scatter(ax, x, y, 36, 'o', ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', color, ...
        'LineWidth', markerLineWidth);
else
    scatter(ax, x, y, 36, 'o', ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', color, ...
        'LineWidth', markerLineWidth, ...
        'DisplayName', labelText);
end
end
