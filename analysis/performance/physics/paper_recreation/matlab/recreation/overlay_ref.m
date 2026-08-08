function plotted = overlay_ref(ax, refDir, filename, color, labelText, markerLineWidth, asDashedLine)
%OVERLAY_REF  Digitized paper data: open-circle markers, or a dashed line.

if nargin < 5
    labelText = '';
end
if nargin < 6
    markerLineWidth = 1.5;
end
if nargin < 7
    asDashedLine = false;
end

[x, y] = load_ref_csv(refDir, filename);
plotted = ~isempty(x);
if ~plotted
    return
end

hold(ax, 'on');
if asDashedLine
    args = {'--', 'Color', color, 'LineWidth', markerLineWidth};
else
    args = {'o', 'LineStyle', 'none', 'MarkerSize', 6, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', color, ...
        'LineWidth', markerLineWidth};
end
if isempty(labelText)
    args = [args, {'HandleVisibility', 'off'}];
else
    args = [args, {'DisplayName', labelText}];
end
plot(ax, x, y, args{:});
end
