function [res] = mat_inprod(A,B)
	% fast computation of trace(A*B')
	res = sum(sum(A.*B));
end