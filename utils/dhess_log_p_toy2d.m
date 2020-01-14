function [dlog_p, hess_p] = d_hess_log_p_dw2(X_in)
	[N,d] = size(X_in);
	hess_p = zeros([d,d,N]);
	X = X_in;
	norm_X = sqrt(sum(X.^2,2));
	X1 = X(:,1);
	Z2 = zeros(N,1);
	expX11 = exp(-2*(X1-3).^2);
	expX12 = exp(-2*(X1+3).^2);
	dexpX1 = 4*((X1-3).*expX11+(X1+3).*expX12)./(expX11+expX12);
	dlog_p = -4*X./norm_X.*(norm_X-3)-[dexpX1,Z2];
	for i = 1:N
		X = X_in(i,:);
		norm_X = norm(X);
		X1 = X(1);
		X2 = X(2);
		Z1 = exp(-2*(X1-3)^2);
		Z2 = exp(-2*(X1+3)^2);
		h11 = 4*(1-3*X2^2/norm_X^3)+4-576*Z1*Z2/(Z1+Z2)^2;
		h12 = 12*X1*X2/norm_X^3;
		h22 = 4*(1-3*X1^2/norm_X^3);
		hess_p(:,:,i) = -[h11,h12;h12,h22];
	end
end