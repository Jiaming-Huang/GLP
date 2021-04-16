# GLP

This repository contains the code and data for “Group Local Projection”. You can find the latest draft and supplemental materials on my [webpage](https://sites.google.com/view/jiaminghuang/research).

Feel free to contact me ([jiaming.huang@barcelonagse.eu](jiaming.huang@barcelonagse.eu)) if you have any doubts on implementing the codes.

## Replication

For the simulation study, the results can be replicated using `SIM_KnownGroupNum.m` (Section 6.2), `SIM_UnknownGroupNum.m` (Section 6.3) and `SIM_NoTrueGroup.m` (Section 6.4). The folder `supp` contains additional exercises in the supplemental material (Section S1), and the multinomial logit model in Section S2.3.

For the empirical application, `EMP_Main.m` gives the baseline results (individual LP-IV, panel LP-IV and the GLP). `EMP_FAVAR.m` shows the FAVAR results. They both take `EMP_data.mat` as input.

The FAVAR application involves two additional folders: 

1. fred: used to estimate the factor model, which is borrowed from the FRED-MD package by McCracken and Ng (2016). I modify the `factors_em.m` file to include the Onatski (2010) criterion. 

2. svar: used to estimate the SVAR-IV model after factor estimation. Here I use the proxy-SVAR upackage provided by Mertens and Montiel-Olea (2018). Alternative inference methods are included (wild bootstrap, delta method and Jentsch Lunsford MBB)

Auxiliary functions in the routine folder are:

- `DGP.m`, `DGP_no_group.m` and `DGP_CompareZ.m`: generate simulated data

- `GroupLPIV.m`: general function for the GLP estimation
  - `GroupLPIV_Infeasible.m`: infeasible GLP in which the latent group is known --this is essentially group-by-group panel LP-IV
  - `GroupLPIV_Sim_Known_Group.m`: GLP that estimates only for a given number of groups G
  - `GroupLPIV_Sim_Unknown_Group.m`: the same as `GroupLPIV.m` except that I remove the inference (as is not reported in simulations)
  
- `eval_GroupLPIV.m`: evaluate the performance of the GLP
  - `getRMSE.m`: an auxiliary function that computes RMSE given the estimated IRs and the true
  
- `ind_LP.m`: individual LP-IV, it can be used also for non-IV case
  - `ind_HAC.m`: HAC standard errors in the individual LP-IV model
  
- `lag.m`: compute lags of variables

- `panel_LP.m`: panel LP-IV, also works for non-IV case

- `params.m`: parameter values in simulation designs

- `prepare_GLP.m`: prepare data to be used as input for `GroupLPIV.m`

- `resid.m`: Compute whether the iteration converges

- `transx.m`: transform variables (comparable to `transxf.m` in the factor estimation package)

## For your own application

You need to:

1. Prepare your data (__balanced panel__) in long format

2. Specify your model (see below an example):

```matlab
FE = 1;                % unit fixed effects - 1; random effects - 0
par.y_idx = 1;         % dependent variable
par.x_idx = 2;         % policy variable (scalar)
par.w_idx = [3 4 5];   % external controls
par.z_idx = [6];       % instrument (now the code is written only for scalar instrument, modification is easy)
par.nylag = 4;         % specify the number of lags, for y x w and z
par.nxlag = 4;
par.nwlag = 4;
par.nzlag = 4;
par.horizon = 24;      % horizons
par.nwtrunc = 25;      % truncation order for Newey-West standard erors (for individual LP-IV)
par.start = '1975-01-01';
par.end   = '2007-12-01';
reg = prepare_GLP(data, par); % now you have your reg (struct variable) 
```

3. Supply the data struct to `GroupLPIV.m`. Before that, it is recommended to run individual LP-IV (to get initial guesses etc), you can otherwise specify your own guesses (and weight matrix).

```matlab
[b_id, se_id, F_id] = ind_LP(reg);  % individual LP-IV

Gmax   = 10;     % maximal number of groups
nInit  = 100;    % number of initializations
bInit  = b_id;   % this is the output from individual LP-IV, from which we can draw initial guesses
weight = se_id;  % this is the weight matrix (I show here the case with L=K=1)
inference = 3;   % inference method: fixed T inference (3), post-GLP inference (2), large T inference (1) 
[Gr_EST, GIRF, GSE, IC] = GroupLPIV(reg, Gmax, nInit, bInit, weight, FE, inference);

```


## References
Huang, J. (2021). Group Local Projections.

McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. Journal of Business & Economic Statistics, 34(4), 574-589.

Mertens, K., & Montiel Olea, J. L. (2018). Marginal tax rates and income: New time series evidence. The Quarterly Journal of Economics, 133(4), 1803-1884.

Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. The Review of Economics and Statistics, 92(4), 1004-1016.
