function [f,g] = MLPclassifciationLoss(w,X,y,nHidden,nLabels,lambda,dropout,reg,bias)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);


%pick regularization
if reg == 2 
    f = lambda/2*w'*w;   
elseif reg == 1
    f= lambda*norm(w,1);
else    
    f= 0;
end


if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output

%Inverted dropout
if dropout > 0
    drop2 = binornd(1,1-dropout,nHidden(2),1)/dropout;
    %drop3 = binornd(1,1-dropout,nHidden(3),1)/dropout;
else 
    drop2 = ones(nHidden(2),1);
    %drop3 = ones(nhidden(3));
end
    
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        if bias == 1
            ip{h}(end) = 1; %add bias at each layer  
        end
        fp{h} = tanh(ip{h});
        if h == 2 %dropout second layer with inverted dropout
            fp{h} = fp{h}.*drop2';
        end
%         if h == 3 %dropout third layer with inverted dropout
%             fp{h} = fp{h}.*drop3';
%         end
        
    end
    yhat = fp{end}*outputWeights;
    
    relativeErr = yhat-y(i,:);
    f = f + relativeErr'*relativeErr;
    
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}';
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                backprop(c,:) = err(c)*(sech(ip{end}).^2.*outputWeights(:,c)');
                gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:);
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                if h == 1
                    backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2.*drop2'; %dropout
%                 elseif h==2
%                     backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2.*drop3'; %dropout
                else
                     backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                end
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
        else
           % Input Weights
            for c = 1:nLabels
                gInput = gInput + err(c)*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
    
    if reg == 2 %regularization
        g = g + lambda*w;
    elseif reg == 1 || w ~= 0 %l1 reg
        g = g + lambda*sign(w);    
    end
end
