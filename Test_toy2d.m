%%%---------------------------------------------%%%
% This implements the 2D toy example in the numerical experiment.
%%%---------------------------------------------%%%
addpath('./utils');
addpath('./solvers');

p = @(X ) exp(-2*(sqrt(sum(X.^2,2))-3).^2)...
    .*(exp(-2*(X(:,1)-3).^2)+exp(-2*(X(:,1)+3).^2));

xnum = 100; ynum = 100;
xmin = -4; ymin = -4;
xmax = 4; ymax = 4;
x = ([1:xnum]/xnum-0.5)*(xmax-xmin)+(xmax+xmin)/2;
y = ([1:ynum]/ynum-0.5)*(ymax-ymin)+(ymax+ymin)/2;
[X,Y] = meshgrid(x,y);
Z = zeros(xnum,ynum,2);
Z(:,:,1)=X;
Z(:,:,2)=Y;
Z_aux = reshape(Z,[xnum*ynum,2]);
Z_ans = p(Z_aux);
Z_plot = reshape(Z_ans,[xnum,ynum]);

rng(2);
N = 200;
X_aux = randn([N,1]);
X_init = randn([N,2])+[0,10];
dlog_p = @dlog_p_toy2d;
dhess_log_p = @dhess_log_p_toy2d;

iters = [5,10,20,40];
%iter = 200;

for i = 1:length(iters)
	iter = iters(i);

	opts1 = struct('tau',0.1,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
	[Xout1,out1] = WGF_m(X_init, dlog_p, opts1);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout1(:,1);
	Yp = Xout1(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	set(gcf,'position',[0,0,640,640]);
	alpha(hp,0.9);
	hold off;
	title(strcat('WGF Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy2d/WGF_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);

	opts2 = struct('tau',0.2,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'lbd',0.5);
	[Xout2,out2] = WNewton_aff_diag(X_init, dhess_log_p, opts2);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout2(:,1);
	Yp = Xout2(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	hold off;
	set(gcf,'position',[0,0,640,640]);
	alpha(hp,0.9);
	title(strcat('WNewton Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy2d/Newton_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);

	opts2 = struct('tau',0.2,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'lbd',0.5);
	[Xout2,out2] = WNewton_aff_mod(X_init, dhess_log_p, opts2);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout2(:,1);
	Yp = Xout2(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	hold off;
	set(gcf,'position',[0,0,640,640]);
	alpha(hp,0.9);
	title(strcat('mWNewton Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat('./result/toy2d/mNewton_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
end



