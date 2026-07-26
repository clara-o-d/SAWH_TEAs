function [x, y] = load_ref_csv(refDir, filename)
%LOAD_REF_CSV  Load digitized (x, y) reference points from a CSV file.

fpath = fullfile(refDir, filename);
if ~isfile(fpath)
    x = [];
    y = [];
    return
end

data = readmatrix(fpath);
if isempty(data)
    x = [];
    y = [];
    return
end

if size(data, 2) < 2
    x = data(:);
    y = nan(size(x));
else
    x = data(:, 1);
    y = data(:, 2);
end

mask = ~(isnan(x) | isnan(y));
x = x(mask);
y = y(mask);
end
