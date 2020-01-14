function [dlog_p,hess_p] = dlog_p_Bayesian(theta, X, Y, batchsize, a0, b0)
%%%%%%%
% Output: First order derivative of Bayesian logistic regression. 
        
% The inference is applied on posterior p(theta|X, Y) with theta = [w, log(alpha)], 
% where p(theta|X, Y) is the bayesian logistic regression 
% We use the same settings as http://icml.cc/2012/papers/360.pdf

% When the number of observations is very huge, computing the derivative of
% log p(x) could be the major computation bottleneck. We can conveniently
% address this problem by approximating with subsampled mini-batches

% Input:
%   -- theta: a set of particles, M*d matrix (M is the number of particles)
%   -- X, Y: observations, where X is the feature matrix and Y contains
%   target label
%   -- batchsize, sub-sampling size of each batch;batchsize = -1, calculating the derivative exactly
%   -- a0, b0: hyper-parameters
%%%%%%%

[N, ~] = size(X);  % N is the number of total observations

if nargin < 4; batchsize = min(N, 100); end % default batch size 100
if nargin < 5; a0 = 1; end
if nargin < 6; b0 = 0.01; end

if batchsize  > 0
    ridx = randperm(N, batchsize);
    X = X(ridx,:); Y = Y(ridx,:);  % stochastic version
end

w = theta(:, 1:end-1);  %logistic weights
alpha_ = exp(theta(:,end)); % the last column is logalpha
D = size(w, 2);

wt = (alpha_/2).*(sum(w.*w, 2));
y_hat = 1./(1+exp(-X*w'));

m = size(theta,1);

dw_data = ((repmat(Y,1,m)+1)/2 - y_hat)' * X; % Y \in {-1,1}
dw_prior = - repmat(alpha_,1,D) .* w;
dw = dw_data * N /size(X,1) + dw_prior; %re-scale

dalpha = D/2 - wt + (a0-1) - b0.*alpha_ + 1;  %the last term is the jacobian term

dlog_p = [dw, dalpha]; % first order derivative 

hess_p = zeros([D+1,D+1,m]);

alpha_w = alpha_.*w;

hess_p(D+1,D+1,:) = -wt-b0*alpha_;
hess_p(1:D,D+1,:) = -alpha_w';
hess_p(D+1,1:D,:) = -alpha_w';
for i = 1:m
	y_hat_tmp = y_hat(:,i);
	tmp1 = (y_hat_tmp.*X)'*((1-y_hat_tmp).*X);
	hess_p(1:D,1:D,i) = -tmp1* N /size(X,1)-alpha_(i)*eye(D);
end

end

