function PrintFigure(filename, alpha)
%Save the current figure with slide-style dimensions and 600 dpi TIFF output.
%
%  PrintFigure('figure3b')          % writes figure3b.tif at alpha = 0.7
%  PrintFigure('figure3b', 1.0)     % full 5×4 inch panel

if nargin < 2
    alpha = 0.7;
end

set(gcf,'color','w');
fig = gcf;
fig.PaperUnits = 'inches';
widthIn = 5 * alpha;
heightIn = 4 * alpha;
fig.PaperPosition = [0 0 widthIn heightIn];
fig.PaperSize = [widthIn heightIn];
drawnow;
print(filename,'-dtiff','-r600','-noui')
end
