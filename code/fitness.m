 function [model] = fitness(X,y, Xvalid, yvalid, nHidden, params)
[n,d] = size(X);
nLabels = max(y);
t = size(Xvalid,1);

%Parameters

%Defaults 
if params == 0
    alpha = 10e-3; %intialize alpha
    beta = 0.9; %for momentum
    bias = 1; % turn bias on or off
    reg = 2; % Regularization 2 for l2, 1 for l1, 0 for off
    lambda = 0.01; % for regularization
    mdecay = 0.8; %multiplicative alpha decay parameter
    dropout = 0; %probability of dropout
   % maxIter = 50000; %maximum number of iterations
else
    
    alpha = params(1); %intialize alpha
    beta =  params(2); %for momentum
    bias = params(3); % turn bias on or off
    reg = params(4); % Regularization 2 for l2, 1 for l1, 0 for off
    lambda = params(5); % for regularization
    mdecay = params(6); %multiplicative alpha decay parameter
    dropout = params(7); %probability of dropout
    maxIter = params(8); %maximum number of iterations
end

yExpanded = linearInd2Binary(y,nLabels);

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient


funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels,lambda,dropout,reg,bias);

%initialize

i = ceil(rand*n);
[f,g] = funObj(w,i);
u =0;
g_n=g;


for iter = 1:maxIter
%     if mod(iter-1,round(maxIter/20)) == 0
%         yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
%         fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
%     end

        %anneal step size
       if mod(iter-1,round(maxIter/10)) == 0
        alpha = alpha*mdecay; %update alpha
       end
       %stepSize = alpha/(1+iter/5000);
    
       
       
%Updates

   w_updated = w -alpha*g_n; 
    i = ceil(rand*n);
   [f_n,g_n] = funObj(w_updated + u,i);  %Nestorov
    u = beta*(w_updated - w);
    w = w_updated + u; %Momentum 

end
yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);

model.fitness = (sum(yhat==yvalid)/t);
model.w = w;

