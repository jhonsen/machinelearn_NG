function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

rejected_X1 = X(find(y==0), 1);
rejected_X2 = X(find(y==0), 2);

admitted_X1 = X(find(y==1), 1);
admitted_X2 = X(find(y==1), 2);


plot(admitted_X1, admitted_X2,'k+', 'Linewidth',2,'MarkerSize',7);
hold on
plot(rejected_X1, rejected_X2,'ko', 'MarkerFaceColor','y','MarkerSize',7);

clearvars rejected_X1 rejected_X2 admitted_X1 admitted_X2
% =========================================================================



hold off;

end
