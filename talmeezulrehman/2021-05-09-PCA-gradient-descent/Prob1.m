format compact; clear; clc; close all;
load face.mat;
%% Preprocessing
mu = mean(X, 2); % complete this part; find the mean of images in X
Y = X - mu; % centered data matrix

%% Perform dimensional reduction via PCA
k = 350;
[U,S,V] = svd(Y); % perform SVD for data matrix Y
Vk = V(:,1:k); % Complete this part; find the first k principal components
X_pca = Y*(Vk*Vk')+mu; % Complete this part; find X_pca

%% Visualization and comparison
I = X(1,:); % Complete this part; pick ANY image (row) from X
I = reshape(I,[112, 92]); % reshape image to the original dimension
I_pca = X_pca(1,:); % Complete this part; pick the corresponding image after PCA
I_pca = reshape(I_pca,[112, 92]); % reshape image to the original dimension

figure(1)
subplot(1,2,1)
imshow(I); 
title('Original','FontSize',16);

subplot(1,2,2)
imshow(I_pca); 
title(['PCA, k=' num2str(k)],'FontSize',16);
