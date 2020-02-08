function [X1, y1,  X2, y2] = splitdata(X, y, ratio)

% initializing variables
num_cols = size(X, 1);
unique_labels = unique(y);
num_classes = size(unique_labels);
d = [1:num_cols]';

X1 = [];
y1= [];

% randomizing and splitting algorithm
for i = 1:num_classes
    current_label = find(y == unique_labels(i));
    if isempty(current_label) 
        continue;
    end
    size_current_label = length(current_label);
    rp = randperm(size_current_label); 
    rp_ratio = rp(1:floor(size_current_label * ratio));
    ind = current_label(rp_ratio);
    X1 = [X1; X(ind, :)];
    y1 = [y1; y(ind, :)];
    d = setdiff(d, ind);
end

X2 = X(d, :);
y2 = y(d, :);

end
