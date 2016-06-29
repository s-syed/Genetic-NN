clear all
load digits.mat
[n,d] = size(X);
nLabels = max(y);
t = size(Xvalid,1);
t2 = size(Xtest,1);
lambda = .1; %L2 Regularization
l = 3; % number of layers
maxIter = 20000; % number of iterations for fitness
cr = 0.25; %crossover rate
mr = 0.1; %mutation rate
gen_max = 100; %maximum generation

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Initial population
gen = 0;
P = 15;
Pop = ceil(200*rand(P,l));
fit = ones(P,1);

while gen <gen_max
    gen
    Pop
    % Find Fitness
    tic
    for i= 1:P
        model = fitness(X,y, Xvalid, yvalid, Pop(i,:), maxIter);
        fit(i) = model.fitness;
    end
    toc
    
    fit_tot = sum(fit);
    fit
    fit_avg(gen+1) = fit_tot/P;
    fit_avg(gen+1)
    
    % Plot our progress      
    semilogy([0:gen],fit_avg);
    pause(.1);
    
    % Selection
    fit = exp(fit)-1;
    fit_tot = sum(fit);
    Survival_prob = fit/fit_tot;
    cum = cumsum(Survival_prob);

    Pop_new = Pop;
    R = rand(P,1);
    for i = 1:P;
        Pop_new(i)= Pop(sum(R(i)>[0;cum]));
    end

    % Crossover
    R = rand(P,1)<=cr;
    pos = find(R); %position of members of population going to undero crossover
    child = ones(1,l);
    if length(pos)>1
        Par = Pop_new(pos(1),:);
        for i = 1:length(pos)
            k = ceil((l-1)*rand); %splicing point
            Par1 = Pop_new(pos(i),:);
            if i<length(pos)     
                Par2 = Pop_new(pos(i+1),:);
            else
                Par2 = Par;
            end
            child(1:k) = Par1(1:k);
            child(k+1:l) = Par2(k+1:l);
            Pop_new(i,:) = child; 
        end
    end

% mutation;

    for i = 1:P
        for j=1:l
            if rand<= mr
            Pop_new(i,j)=ceil(200*rand);
            end
        end
    end
    Pop = Pop_new;
    gen = gen+1;
end

Pop
% Find final fitness
    for i= 1:P
        model = fitness(X,y, Xvalid, yvalid, Pop(i,:), maxIter);
        fit(i) = model.fitness;
    end
    fit
    fit_tot = sum(fit);
    fit_avg(gen+1) = fit_tot/P;
    fit_avg(gen+1)
        % Plot our progress      
    semilogy([0:gen],fit_avg);
    pause(.1);