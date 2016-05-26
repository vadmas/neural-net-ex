function [model] = vanillaNeuralNet(X,y,lambda,epsilon,hiddenNodes,iteration)

% Add bias variable
[n,d] = size(X);
[outputDim,~] = size(unique(y));
W1 = rand(d,hiddenNodes) / sqrt(d);
b1 = zeros(1,hiddenNodes);
W2 = rand(hiddenNodes,outputDim) / sqrt(hiddenNodes);
b2 = zeros(1,outputDim);

model.W1 = W1;
model.b1 = b1;
model.W2 = W2;
model.b2 = b2;
model.lambda = lambda;
for iter = 1:iteration
	[probs, z1, a1, z2] = forwardProp(model,X);

	% Back prop
	idx = sub2ind(size(probs),[1:n]',y+1);
	delta3 = probs;
	delta3(idx) = delta3(idx) - 1;
	dW2 = a1'*delta3;
	db2 = sum(delta3);
	delta2 = delta3*W2' .* (1 - a1.^2);
	dW1 = X' * delta2;
	db1 = sum(delta2);
	
	% Regularize 
  	dW2 = dW2 + lambda*W2;
    dW1 = dW1 + lambda*W1;

    % Step
	W1 = W1 - epsilon* dW1;
	b1 = b1 - epsilon* db1;
	W2 = W2 - epsilon* dW2;
	b2 = b2 - epsilon* db2;

	% Save
	model.W1 = W1;
	model.b1 = b1;
	model.W2 = W2;
	model.b2 = b2;

	if mod(iter,1000) == 0 
		fprintf('Training iteration = %d, validation error = %f\n',iter,loss(model,X,y));
	end

end
model.predict = @predict;
end

%% loss: Calculate loss function
function [loss] = loss(model, X, y)
	[n,d] = size(X);
	[outputDim,~] = size(unique(y));
	W1 = model.W1;
	b1 = model.b1;
	W2 = model.W2;
	b2 = model.b2;
	lambda = model.lambda;

	[probs, z1, a1, z2] = forwardProp(model, X);
	idx = sub2ind(size(probs),[1:n]',y+1);
	loss = sum(-log(probs(idx)));
	% Add regularization to loss
	loss = loss + lambda/2 * (norm(W1,'fro')^2 + norm(W2,'fro')^2);
	% Normalize
	loss = loss / n;
end

%% forwardProp: Propagates a given X through a nn. 
%% Used as a private helper function
function [probs, z1, a1, z2] = forwardProp(model, X)
	W1 = model.W1;
	b1 = model.b1;
	W2 = model.W2;
	b2 = model.b2;
 
	z1 = bsxfun(@plus,X*W1,b1);
	a1 = tanh(z1);
	z2 = bsxfun(@plus,a1*W2,b2);
	expz2 = exp(z2);
	probs = expz2 ./ repmat(sum(expz2,2),1,2);
end

function [yhat] = predict(model,Xhat)
	[probs,~,~,~] = forwardProp(model,Xhat);
	[~,yhat] = max(probs,[],2);
	% Classes are 0,1 not 1,2
	yhat = yhat - 1;
end
