# Information Newton's flow

This program implements the numerical experiement part of __Information Newton's flow: second-order optimization method in probability space__. 



## Abstract

We introduce a framework for Newton's flows in probability space with information metrics, named information Newton's flows. Here two information metrics are considered, including both the Fisher-Rao metric and the Wasserstein-2 metric. Several examples of information Newton's flows for learning objective/loss functions are provided, such as Kullback-Leibler (KL) divergence, Maximum mean discrepancy (MMD), and cross entropy. The asymptotic convergence results of proposed Newton's methods are provided. A known fact is that overdamped Langevin dynamics correspond to Wasserstein gradient flows of KL divergence. Extending this fact to Wasserstein Newton's flows of KL divergence, we derive Newton's Langevin dynamics. We provide examples of Newton's Langevin dynamics in both one-dimensional space and Gaussian families. For the numerical implementation, we design sampling efficient variational methods to approximate Wasserstein Newton's directions. Several numerical examples in Gaussian families and Bayesian logistic regression are shown to demonstrate the effectiveness of the proposed method. 



## Reproduction

- For the toy example: Directly run `Test_toy1d.m` and `Test_toy2d.m`. Figures will be saved under `./result/toy1d/` and `./result/toy2d/` 

- For the Gaussian examples:  run `Test_Gauss.m`. Figures will be saved under `./result/Gauss`

- For the Bayesian logistic regression:  

  First download the covertype dataset from

  https://github.com/DartML/Stein-Variational-Gradient-Descent

  Place `covertype.mat` under the folder `./data/`

  Then, run `Bayesian_MED.m` to output datafiles. Run `Bayesian_plot.m` to plot the figures. You can use our existing data file to plot figures. 



## Feedback

If you have any questions or comments, feel free to send me an [email](zackwang24@pku.edu.cn). 