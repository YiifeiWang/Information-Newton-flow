function [S] = Gaussian_CG(Sigma,A,T,opts)
	% implement the CG method in Gaussian families
	[n,~] = size(Sigma);
	if nargin < 4; opts = []; end
	if ~isfield(opts, 'max_iter'); opts.max_iter = 100; end
	if ~isfield(opts, 'crit'); opts.crit = 1; end
	if ~isfield(opts, 'tol');  opts.tol = 1e-8; end
	if ~isfield(opts, 'S'); opts.S = eye(n); end
	if ~isfield(opts, 'record'); opts.record = 0; end
	if ~isfield(opts, 'itPrint'); opts.itPrint = 0; end

	max_iter = opts.max_iter;
	record = opts.record;
	itPrint = opts.itPrint;

	if record == 1
	    % set up print format
	    if ispc; str1 = '  %10s'; str2 = '  %6s';
	    else     str1 = '  %10s'; str2 = '  %6s'; end
	    stra = ['%5s',str2,str2,str2,str1, str2,'\n'];
	    str_head = sprintf(stra, ...
	        'iter', 'H','F_norm','inf_norm', 'pos', 'TBA');
	    str_num = ['%4d | %+2.4e %+2.4e %+2.4e %d %d \n'];
	end

	S = opts.S;
	ASSigma = A*S*Sigma;
	R = ASSigma+ASSigma'+2*S-T;
	P = -R;
	cstop = 0;
	alphak = 0;
	betak = 0;
	trAPP = 0;
	norm_r = mat_inprod(R,R);
	for iter = 1:max_iter
		if record
	        if iter == 1 || mod(iter,20*itPrint) == 0 
	            fprintf('\n%s', str_head);
	        end
	        H = mat_inprod(R,R);
	        if iter == 1 || mod(iter,itPrint) == 0
				fprintf(str_num,iter,H, alphak, betak,trAPP,norm(P-P'));
	        end
		end
		
		trAPP = mat_inprod(A*P,P*Sigma)+mat_inprod(P,P);
		if trAPP>1e8
			P = -R;
		end
		alphak = norm_r/trAPP/2;
		S = S+alphak*P;
		ASSigma = A*S*Sigma;
		norm_rp = norm_r;
		R = ASSigma+ASSigma'+2*S-T;
		norm_r = mat_inprod(R,R);
		switch opts.crit
			case 1
				if (sqrt(norm_r)<opts.tol)	
					cstop = 1;
				end
		end
		if cstop
			break
		end
		betak = norm_r/norm_rp;
		P = -R+betak*P;
	end
