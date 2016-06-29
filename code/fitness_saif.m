function [model] = fitness(X,y, Xvalid, yvalid, nHidden, maxIter)
[n,d] = size(X);
nLabels = max(y);
t = size(Xvalid,1);
lambda = .1;

yExpanded = linearInd2Binary(y,nLabels);

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
stepSize_0 = 1e-2;
lambda = .1;

funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels,lambda);
for iter = 1:maxIter
%     if mod(iter-1,round(maxIter/20)) == 0
%         yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
%         fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
%     end
    stepSize = stepSize_0/(1+iter/5000);
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end
yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);

model.fitness = (sum(yhat==yvalid)/t);
model.w = w;

