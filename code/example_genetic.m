clear all
load digits.mat
[n,d] = size(X);
nLabels = max(y);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

%Genetic Parameters
P = 8; % population size
l = 5th; % number of layers
cr = 0.3; % crossover rate
mr = 0.3; % mutation rate
gen_max = 100; % maximum generations
max_hidden = 250; %maximum number of hidden units per layer

% Initial population
gen = 1;
Pop = ceil(max_hidden*rand(P,l));
fit = ones(P,1);

%Population parameters
params = zeros(8,1); %vector of parameters

    params(1) = 10e-3; %alpha (step size)
    params(2) = 0.9; %beta for momentum
    params(3) = 0; % turn bias on or off
    params(4) = 2; % Regularization 2 for l2, 1 for l1, 0 for off
    params(5) = 0.01; %lambda for regularization
    params(6) = 0.7; %alpha decay 
    params(7) = 0.5; %probability of dropout
    params(8) = 250000; %maximum number of iterations

    %array storing parameters for each neural net in population
    
    
    parameters = zeros(P,length(params)); %allow different parameters for each net
    
   for i = 1:P
    parameters(i,:) = params';
   end
   
   params_index = cell(length(params),1);
   params_index{1} = 'alpha';
   params_index{2} = 'beta';
   params_index{3} = 'bias';
   params_index{4} = 'regular';
   params_index{5} = 'lambda';
   params_index{6} = 'alpha decay';
   params_index{7} = 'dropout';
   params_index{8} = 'iterations';
   
   %ensemble tracking
   ensemble = cell(gen_max,4);
   
    y_ensemble = zeros(t,gen*P);
    y_ensemble_test = zeros(t2,gen*P);
    ensemble_fitness = zeros(gen_max,1);
    

    
   
    
    %Calculate Ensemble
    fit_cut = 0.8; %Only networks above this value will be considered
    c=0; %counter for number of neural networks considered
   
   
while gen <gen_max +1
    gen
    
    %Uncomment line below to show parameters with lables on console.
   
    %param_ouput
    %= [params_index';num2cell(parameters)]
    % Find Fitness
    tic
    
    weights = cell(P,1);
    for i= 1:P
        model = fitness(X,y, Xvalid, yvalid, Pop(i,:), parameters(i,:));
        fit(i) = model.fitness;
        weights{i} = model.w; %store model weights
    end
    toc
    
    fit_tot = sum(fit);
    [fit Pop]
    fit_avg(gen) = fit_tot/P;
    fit_avg(gen)
    fit_max(gen) = max(fit);
    
    %Ensemble tracking
    ensemble{gen,1} = fit;
    ensemble{gen,2} = Pop;
    ensemble{gen,3} = parameters;
    ensemble{gen,4} = weights;
    

    

        for j = 1:P
            if ensemble{gen,1}(j) > fit_cut
                c = c+1; %increase count
                y_ensemble(:,c) = MLPclassificationPredict(ensemble{gen,4}{j},Xvalid,ensemble{gen,2}(j,:),nLabels);
                y_ensemble_test(:,c) = MLPclassificationPredict(ensemble{gen,4}{j},Xtest,ensemble{gen,2}(j,:),nLabels);
            end
        end
   
    
    if c> 0
        
        c
        y_hat = mode(y_ensemble(:,1:c), 2); %extract first c columns
        ensemble_fitness(gen) = (sum(y_hat==yvalid)/t);   %calculate ensemble fitness
        ensemble_fitness(gen)
        
        %calculate test error
        y_hat = mode(y_ensemble_test(:,1:c), 2); %extract first c columns
        sum(y_hat==ytest)/t2 %show test fitness
    else
        c
        ensemble_fitness(gen) = 0;
        ensemble_fitness(gen)
    end
    
    % Plot our progress 
%     plot([1:gen],fit_avg);
%     hold;
%     plot([1:gen],fit_max);
%     hold;
%     plot([1:gen],ensemble_fitness);
%     hold;
%     axis([0 gen 0 1]);
%     
    p1 = subplot(1,2,1);
    plot(p1,[1:gen],fit_avg);
    hold;
    plot(p1,[1:gen],fit_max);
    hold;
    title('Average/Max Generation Fitness');
    
    p2 =subplot(1,2,2);
    plot(p2,[1:gen],ensemble_fitness(1:gen,1));
    title('Ensemble Average');

    axis([p1,p2],[0 gen 0 1]);
    pause(.01);
    
    % Selection
    %fit = exp(fit)-1;
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
            Pop_new(i,j)=ceil(max_hidden*rand);
            end
        end
    end
    Pop = Pop_new;
    gen = gen+1;
end

Pop
% Find final fitness
    for i= 1:P
        model = fitness(X,y, Xvalid, yvalid, Pop(i,:), parameters(i,:));
        fit(i) = model.fitness;
    end
    fit
    fit_tot = sum(fit);
    fit_avg(gen) = fit_tot/P;
        % Plot our progress      
    semilogy([1:gen],fit_avg);
    pause(.1);
