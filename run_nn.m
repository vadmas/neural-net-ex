% Clear variables and close figures
clear all
close all

% Load data
load data/labels.mat % Loads X and y
load data/moons.mat % Loads X and y
X = moons;
y = labels';

lambda = 0.01;
epsilon = 0.01;
hiddenNodes = 10;
% iter = 20000;
iter = 2000;

model = vanillaNeuralNet(X,y,lambda,epsilon,hiddenNodes,iter);
% Show data and decision boundaries
plot2DClassifier(X,y,model);