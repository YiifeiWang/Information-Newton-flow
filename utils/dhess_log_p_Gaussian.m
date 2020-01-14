function [dlog_p,hess_p] = hess_p_Gaussian(X_in,A)
	[N,d] = size(X_in);
	dlog_p = -X_in*A;
	hess_p = reshape(repmat(-A,1,N),d,d,N);
end