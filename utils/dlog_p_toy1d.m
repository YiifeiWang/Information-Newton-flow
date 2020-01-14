function [dlog_p] = dlog_p_dw1(X)

	dlog_p = -2*X.*(X.^2-1);

end