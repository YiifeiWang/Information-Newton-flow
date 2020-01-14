%%%---------------------------------------------%%%
% This implements the example in Gaussian families.
%%%---------------------------------------------%%%
addpath('./utils');
addpath('./solvers');

rng(3);
d = 100;
aux = randn([d,d]);
A = aux'*aux+1e-3*eye(d);

fprintf('Condition number of W: %.4f\n',cond(A));

dlog_p = @(X) dlog_p_Gaussian(X,A);
tau = 1/max(eig(A))/2;

dhess_log_p = @(X ) dhess_log_p_Gaussian(X,A);

N = 500;
X_init = randn([N,d]);

opts1 = struct('iter_num',50,'ptype',0,'A',A,'record',0,'itPrint',1,'trace',1,'ktype',3,'tau',1);
[~,out1] = WNewton_aff_mod(X_init,dhess_log_p,opts1);

opts2 = struct('iter_num',4e3,'ptype',0,'A',A,'record',0,'itPrint',1e2,'trace',1,'ktype',3,'tau',tau);
[~,out2] = WGF_m(X_init,dlog_p,opts2);

opts3 =struct('iter_num',3e3,'ptype',0,'A',A,'record',0,'itPrint',1e2,'trace',1,'ktype',3, ...
	'restart',1,'strong',0,'tau',tau);
[~,out3] = AIG_SE(X_init,dlog_p,opts3);

opts4 =struct('iter_num',3e3,'ptype',0,'A',A,'record',0,'itPrint',1e2,'trace',1,'ktype',3,'tau',1);
[~,out4] = HALLD_m(X_init,dhess_log_p,opts4);

opts5 =struct('iter_num',11,'ptype',0,'A',A,'record',0,'itPrint',1,'trace',1,'ktype',3,'tau',0.5);
[~,out5] = HALLD_m(X_init,dhess_log_p,opts5);

intervals = ones([10,1])*10;
markers     = {'d-','*-','s-','<-','^-','*-','v-','>-','o-','*-','.-','s-','d-','^-','v-','>-','<-','p-','h-'};
colors = {[0,0,1],[1,0,1],[0,1,0],...     
          [255,71,71]/255,... 
          [0.9,0.7,0.0],...
          [0,101,189]/255,...          
          [17,140,17]/255,...       
          [0.9,0.7,0.0], ...        
          [0,101,189]/255,...       
          [1,1,0],[1,1,1],[1,0,1],[0,1,1],[0,1,0],[0,0,1],[0.0,0.3,0.8],[1,0,0]/255};  


figure(1)
clf;
% WNewton
semilogy_marker(out1.trace.iter,out1.trace.H,markers{1},5,10,colors{1});
% WGF
semilogy_marker(out2.trace.iter,out2.trace.H,markers{2},5,10,colors{2});
% AIG
semilogy_marker(out3.trace.iter,out3.trace.H,markers{3},5,10,colors{3});
% HALLD 1
semilogy_marker(out4.trace.iter,out4.trace.H,markers{4},5,10,colors{4});
% HALLD 0.5
semilogy_marker(out5.trace.iter,out5.trace.H,markers{5},5,10,colors{5});
ylim([1e-12,1e5]);
xlim([0,3e3]);
legend({'mWNewton','WGF','AIG','HALLD 1','HALLD 0.5'},'location','southeast');
xlabel('Iteration');
ylabel('KL divergence');

set(gcf,'position',[0,0,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','.result/Gauss/Gauss_iter.eps');

figure(2)
clf;
% WNewton
semilogy_marker(out1.trace.time,out1.trace.H,markers{1},5,10,colors{1});
% WGF
semilogy_marker(out2.trace.time,out2.trace.H,markers{2},5,10,colors{2});
% AIG
semilogy_marker(out3.trace.time,out3.trace.H,markers{3},5,10,colors{3});
% HALLD 1
semilogy_marker(out4.trace.time,out4.trace.H,markers{4},5,10,colors{4});
% HALLD 0.5
semilogy_marker(out5.trace.time,out5.trace.H,markers{5},5,10,colors{5});
ylim([1e-12,1e5]);
xlim([0,1.8]);
legend({'mWNewton','WGF','AIG','HALLD 1','HALLD 0.5'},'location','southeast');
xlabel('Time');
ylabel('KL divergence');

set(gcf,'position',[0,0,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','.result/Gauss/Gauss_time.eps');