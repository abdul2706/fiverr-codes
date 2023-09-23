function X = compressImg(I,k)
% Input: I: data matrix for image 
%        k: number of colors used for compressing image
% Ouput: X: matrix for compressed image with k colors

I_float = double(I); % convert the data type of data matrix to float
[m,n,c] = size(I_float); % get dimensions of I, c = 3 here (for RGB channels) 

%% Clutering with kmeans
I_reshape = reshape(I_float,[m*n,c]);
% complete this part; call the subroutine 'reshape' to reshape 
% 3-D 'I_float' with dimensions m x n x c into a 2-D matrix 'I_reshape';
% read documentation for 'kmeans' to understand why reshaping 

rng(111); % fix the random seed kmeans algorithm

[idx,Centr] = kmeans(I_reshape,k,"MaxIter",500);
% complete this part; call 'kmeans' to cluster the data in 'I_reshape';
% understand the outputs 'idx' and 'Centr'

fprintf("[idx]\n"); disp(size(idx)); disp(unique(idx));
fprintf("[Centr]\n"); disp(Centr);

%% Get compressed image
X_temp = zeros(size(I_reshape));
% create a zero matrix with same dimension 
% as 'I_reshape' to store the pixel values of compressed image

for i = 1:m*n % there are in total m*n pixels
    X_temp(i,:) = Centr(idx(i),:); % complete this part using 'idx' and 'Centr';
    % RGB value for pixels of compressed image (i.e. assigned centroids)
    % should be stored as rows in 'X_temp'
end

X = reshape(X_temp,[m,n,c]); % complete this part; reshape 'X_temp' back to the orignal shape
X = uint8(X); % convert the data type back to integer

end
