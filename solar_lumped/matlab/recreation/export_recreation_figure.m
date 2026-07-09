function export_recreation_figure(fig, filepathNoExt, alpha, gridSize)
%EXPORT_RECREATION_FIGURE  Save at PrintFigure slide dimensions (600 dpi TIFF).
%
%   export_recreation_figure(fig, 'path/to/figure3', 0.7, [1 3])
%     → 3.5×2.8 in per panel, 1×3 grid → 10.5×2.8 in total
%
%   gridSize is [nrows, ncols]. Defaults to [1 1] for a single panel.

if nargin < 3 || isempty(alpha)
    alpha = 0.7;
end
if nargin < 4 || isempty(gridSize)
    gridSize = [1, 1];
end

set(fig, 'Color', 'w', 'InvertHardcopy', 'off');
widthIn = 5 * alpha * gridSize(2);
heightIn = 4 * alpha * gridSize(1);
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 widthIn heightIn];
fig.PaperSize = [widthIn heightIn];

fprintf('Saving %s (%0.1f × %0.1f in @ 600 dpi)...\n', ...
    filepathNoExt, widthIn, heightIn);
drawnow;
print(fig, filepathNoExt, '-dtiff', '-r600', '-noui');
fprintf('  → %s.tif\n', filepathNoExt);
end
