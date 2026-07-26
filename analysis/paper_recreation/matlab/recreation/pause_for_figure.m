function pause_for_figure(fig)
%PAUSE_FOR_FIGURE  Show the figure and wait for Enter in the command window.

drawnow;
if nargin >= 1 && ~isempty(fig) && ishandle(fig)
    figure(fig);
    fprintf('  [Figure %d] Press Enter in the command window to continue...\n', fig.Number);
else
    fprintf('Press Enter in the command window to continue...\n');
end
pause;
end
