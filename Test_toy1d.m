%%%---------------------------------------------%%%
% This implements the 1D toy example in the numerical experiment.
%%%---------------------------------------------%%%
addpath('./utils');
addpath('./solvers');

rng(1);
N = 100;
X_init = randn([N,1])/10;
dlog_p = @dlog_p_toy1d;
dhess_log_p = @dhess_log_p_toy1d;

toy1d = @(X) exp(-0.5*(X.^2-1).^2);

iters = [2,5,10,20];
X_p = ([0:100]/100-0.5)*5;
Y_p = toy1d(X_p);

for i = 1:length(iters)
	iter = iters(i);

	opts1 = struct('tau',0.01,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
	[Xout1,out1] = WGF_m(X_init, dlog_p, opts1);

	clf
	figure(1);
	Yout1 = toy1d(Xout1);
	plot(X_p,Y_p,'LineWidth',1);
	hold on;
	hp=scatter(Xout1,Yout1,40,'r');
	set(gcf,'position',[0,0,640,320]);
	alpha(hp,0.8);
	hold off;
	
	title(strcat('WGF Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy1d/WGF_iter',mat2str(iter),'_1d.eps');
	print('-depsc',save_path1);

	opts2 = struct('tau',1,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
	[Xout2,out2] = WNewton_aff_diag(X_init, dhess_log_p, opts2);

	clf
	figure(2);
	Yout2 = toy1d(Xout2);
	plot(X_p,Y_p,'LineWidth',1);
	hold on;
	hp=scatter(Xout2,Yout2,40,'r');
	set(gcf,'position',[0,0,640,320]);
	alpha(hp,0.8);
	hold off;
	title(strcat('WNewton Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy1d/Newton_iter',mat2str(iter),'_1d.eps');
	print('-depsc',save_path1);

	opts2 = struct('tau',1,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
	[Xout2,out2] = WNewton_aff_mod(X_init, dhess_log_p, opts2);

	clf
	figure(2);
	Yout2 = toy1d(Xout2);
	plot(X_p,Y_p,'LineWidth',1);
	hold on;
	hp=scatter(Xout2,Yout2,40,'r');
	set(gcf,'position',[0,0,640,320]);
	alpha(hp,0.8);
	hold off;
	title(strcat('mWNewton Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy1d/mNewton_iter',mat2str(iter),'_1d.eps');
	print('-depsc',save_path1);

	opts3 = struct('tau',0.01,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'avg',0);
	[Xout3,out3] = HALLD_m(X_init, dhess_log_p, opts3);

	clf
	figure(3);
	Yout3 = toy1d(Xout3);
	plot(X_p,Y_p,'LineWidth',1);
	hold on;
	hp=scatter(Xout3,Yout3,40,'r');
	set(gcf,'position',[0,0,640,320]);
	alpha(hp,0.8);
	hold off;
	title(strcat('HALLD Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy1d/HALLD_iter',mat2str(iter),'_1d.eps');
	print('-depsc',save_path1);

end



