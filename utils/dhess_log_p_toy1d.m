function [dlog_p,Hess_p] = dlog_p_dw1(X)
	[N,~] = size(X);
	dlog_p = -2*X.*(X.^2-1);
	Hess_p = zeros(1,1,N);
	Hess_p(1,1,:) = -6*X.^2+2;

end