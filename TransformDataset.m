function X = TransformDataset(X)
% TransfromDataset, looks through an array and transforms all the columns
% only containing 2 unique values into binary columns instead.
[~, columns] = size(X);
% Loop to iterate over each column
for i=1:columns
[C,~,ic] = unique(X(:,i),'stable'); % Finding unique values

if size(C,1)==2 %If only two unique values
    X(:,i) = ic-1; % Change to binary (0 & 1)
end
end

end
