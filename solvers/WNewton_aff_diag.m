function [Xs,out] = WNewton_aff_diag(X_init, d_hess_log_p, opts)
%%%---------------------------------------------%%% 
% This implements the (diagonal) affine Wasserstein Newton's method
% 
% Input:
% 		X_init --- initial particle positions, N*d matrix
%		dlog_p --- function handle to compute the derivative of log p(x)
%		opts   --- options structure with fields
%				   tau --- step size
%		      iter_num --- number of iterations
%				 ktype --- the type of the kernel
%        		   ibw --- inverse of the bandwidth of the kernel
%                trace --- whether to trace the information during the process
%						   0: no || 1: yes
%				record --- whether to print the recorded inforamtion during the process
%						   0: no || 1: yes
%			   itPrint --- the interval of printing
%			 	 ptype --- the type of the problem
%						   0: Gaussian
%							  opts.A shall NOT be empty
%						   1: Bayesian logistic regression
%							  opts.X_test, opts.y_test shall NOT be empty
%				 rtype --- option for the kernel bandwidth selection subroutine
%				 h_pow --- parameter for the kernel bandwidth selection subroutine
%				 ttype --- option in decaying tau
%			   tau_dec --- decay rate of the step size
%			   tau_itv --- decay interval of the step size
%
%				  epsl --- parameter for the regularization
%				   lbd --- parameter for the hybrid update
%				    bm --- whether to use OLD as the Wasserstein gradient direction
%						   0: no || 1: yes
%
%
% Author: Yifei Wang, 2020
% 
%%%---------------------------------------------%%%
	tic;
	[N,d] = size(X_init);
	if nargin < 3; opts = []; end
	if ~isfield(opts, 'tau'); opts.tau = 0.1;  end
	if ~isfield(opts, 'iter_num'); opts.iter_num = 1e3; end
	if ~isfield(opts, 'sub_iter_num'); opts.sub_iter_num = 10; end
	if ~isfield(opts, 'ktype');  opts.ktype = 1; end
	if ~isfield(opts, 'ibw');  opts.ibw = -1; end

	if ~isfield(opts, 'trace'); opts.trace = 0; end
	if ~isfield(opts, 'record'); opts.record = 0; end
	if ~isfield(opts, 'itPrint'); opts.itPrint = 1; end
	if ~isfield(opts, 'ptype'); opts.ptype = 1; end 
	if ~isfield(opts, 'rtype'); opts.rtype = 1; end 

	if ~isfield(opts, 'h_pow'); opts.h_pow = d/2+2; end

	if ~isfield(opts, 'ttype'); opts.ttype = 1; end
	if ~isfield(opts, 'tau_dec'); opts.tau_dec = 1; end
	if ~isfield(opts, 'tau_itv'); opts.tau_itv = 100; end

	if ~isfield(opts, 'lbd'); opts.lbd = 0; end
	if ~isfield(opts, 'epsl'); opts.epsl = 0; end
	if ~isfield(opts, 'bm'); opts.bm = 0; end

	if ~isfield(opts, 'beta'); opts.beta = 1; end

	tau = opts.tau;
	iter_num = opts.iter_num;
	ktype = opts.ktype;
	ibw = opts.ibw;

	sub_iter_num = opts.sub_iter_num;


	record = opts.record;
	itPrint = opts.itPrint;
	ptype = opts.ptype;

	lbd = opts.lbd;
	epsl = opts.epsl;

	X = X_init;

	trace_ = [];
	if opts.trace
		switch ptype
			case 0
				trace_.H = zeros(floor(iter_num/itPrint)+1,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+1,1);
			case 1
				trace_.test_acc = zeros(floor(iter_num/itPrint)+1,1);
				trace_.test_llh = zeros(floor(iter_num/itPrint)+1,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+1,1);
				trace_.time = zeros(floor(iter_num/itPrint)+1,1);
		end
	end

	if opts.ptype == 0
		A = opts.A;
		logdetA = log(det(A));
	end

	if record == 1
	    % set up print format
	    if ispc; str1 = '  %10s'; str2 = '  %6s';
	    else     str1 = '  %10s'; str2 = '  %6s'; end
	    stra = ['%5s',str2,str2,str2,str1, str2,'\n'];
	    str_head = sprintf(stra, ...
	        'iter', 'TBA','TBA','TBA', 'TBA', 'TBA');
	    str_num = ['%4d | %+2.4e %+2.4e %+2.4e %2.1e %2.1e \n'];
	end
	
	eva_time = 0;
	sb = 0;

	I = eye(d);
	for iter = 1:iter_num
		b_diff = 0;
		if opts.trace && (iter == 1 || mod(iter,itPrint)==0)
			switch ptype
				case 0
					tmp1 = toc;
					Sigma = cov(X);
					if cond(Sigma)<1e-12;
						break
					end
					tmp = A*Sigma;
					H = (-d-log(det(Sigma))-logdetA+trace(tmp))/2;
					eva_time = eva_time+toc-tmp1;
					trace_.H(floor(iter/itPrint)+1) = H;
					trace_.iter(floor(iter/itPrint)+1) = iter;
					trace_.time(floor(iter/itPrint)+1) = toc-eva_time;
				case 1
					tmp1 = toc;
					[test_acc, test_llh] = bayeslr_evaluation(X, opts.X_test, opts.y_test);
					eva_time = eva_time+toc-tmp1;
					trace_.test_acc(floor(iter/itPrint)+1) = test_acc;
					trace_.test_llh(floor(iter/itPrint)+1) = test_llh;
					trace_.iter(floor(iter/itPrint)+1) = iter;
					trace_.time(floor(iter/itPrint)+1) = toc-eva_time;
				case 2
					trace_(floor(iter/itPrint)+1) = ibw;
			end
			if record
		        if iter == 1 || mod(iter,20*itPrint) == 0 
		            fprintf('\n%s', str_head);
		        end
		        if iter == 1 || mod(iter,itPrint) == 0
		        	switch ptype
						case 0
							fprintf(str_num,iter,H, min(eig(Sigma)), max(eig(Sigma)),ibw,0);
						case 1
							fprintf(str_num,iter,test_acc, test_llh, 0,0,0);
						case 2
							fprintf(str_num,iter,ibw, b_diff, 0,0,0);
					end
		        end
			end
		end

		switch opts.ttype
			case 1
				tau = opts.tau*opts.tau_dec^(floor(iter/opts.tau_itv));
			case 2
				tau = opts.tau/iter^opts.tau_dec;
		end
		[grad, Hess] = d_hess_log_p(X);
		kopts = struct('iter',iter,'rtype',opts.rtype,'ibw',ibw,'h_pow',opts.h_pow,'tau',tau);
		[xi, ibw_out] = dlog_p_prac(X,ktype,kopts);
		ibw = ibw_out;
		v = xi-grad;
		Hess_mod = -Hess + epsl*reshape(repmat(I,1,N),d,d,N);
		Fsum = sum(Hess_mod,3);
		vsum = sum(v);

		H = zeros([2*d,2*d]);

		t = zeros([1,2*d]);

		for i =1:N
			tmp =diag(X(i,:))*squeeze(Hess_mod(:,:,i));
			H(1:d,1:d) = H(1:d,1:d)+tmp*diag(X(i,:));
			H(1:d,d+1:2*d) = H(1:d,d+1:2*d)+tmp;
			t(1:d) = t(1:d)+X(i,:).*v(i,:);
		end
		H(d+1:2*d,1:d) = transpose(H(1:d,d+1:2*d));
		H(d+1:2*d,d+1:2*d) = Fsum;
		H = H/N;
		H(1:d,1:d) = H(1:d,1:d)+I;

		t(d+1:2*d) = vsum;
		t = t/N;

		t = reshape(t,[],1);

		sb = -H\t;

		sb = reshape(sb,1,[]);

		s = sb(1:d);
		b = sb(d+1:2*d);
		
		if opts.bm
			X = X+tau*(X.*s+b+lbd*grad)+sqrt(tau*lbd)*randn(size(X));
		else
			X = X+tau*(X.*s+b-lbd*v);
		end

		
	end

	Xs = X;
	out = [];
	if opts.trace
		out.trace = trace_;
	end
end