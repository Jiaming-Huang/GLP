# GLP

This repository contains the code and data for “Group Local Projection”. You can find the latest draft and supplemental materials on my [webpage](https://sites.google.com/view/jiaminghuang/research).

Feel free to contact me ([jiaming.huang@barcelonagse.eu](jiaming.huang@barcelonagse.eu)) if you have any doubts on implementing the codes.

## Replication

For the simulation study, the results can be replicated using `SIM_KnownG0.m` (Section 6.2), `SIM_UnknownG0.m` (Section 6.3) and `SIM_NoTrueGroup.m` (Section 6.4). The folder `supp` contains additional exercises in the supplemental material (Section S1), and the multinomial logit model in Section S2.3.

For the empirical application, `EMP_Main.m` gives the baseline results (individual LP-IV, panel LP-IV and the GLP). This is compared with ad hoc grouping criterion (`EMP_Adhoc.m`) and FAVAR (`EMP_FAVAR.m`). They both take `EMP_data.mat` as input.

The FAVAR application involves two additional folders: 

1. fred: used to estimate the factor model, which is borrowed from the FRED-MD package by McCracken and Ng (2016). I modify the `factors_em.m` file to include the Onatski (2010) criterion. 

2. svar: used to estimate the SVAR-IV model after factor estimation. Here I use the proxy-SVAR upackage provided by Mertens and Montiel-Olea (2018). Alternative inference methods are included (wild bootstrap, delta method and Jentsch Lunsford MBB)

Auxiliary functions in the routine folder are:

- `DGP.m`, `DGP_NoGroup.m` and `DGP_FDAH.m`: generate simulated data

- `GLP.m`: general function for the GLP estimation
  - `HAC4d.m`: compute HAC robust variance estimator for 4-d matrices; see equation (17)
  - `GLP_SIM_Infeasible.m`: infeasible group-by-group panel LP-IV in which the latent group is known --this is essentially xtivreg2 for each group; related is `GLP_SIM_Infeasible1.m` where we know the true group assignment but still use our GMM objective function (see SM S1.6)
  - `GLP_SIM_KnownG0.m`: GLP with known number of groups G0 (Section 6.2)
  - `GLP_SIM_UnknownG0.m`: GLP that runs for Ghat=1,...Gmax, and select the number of groups by IC; see equation (28)-(29) (I remove the inference as is not reported in simulations)
  - `GLP_SIM_KnownG0_Inference.m`: same as `GLP_SIM_KnownG0.m` except that we return results for both large T and small T inference
  - `GLP_SIM_KnownG0_Weight.m`: the same as `GLP_SIM_KnownG0.m` except that we return results for unit-and-horizon specific weighting and mixed weighting (see SM S1.3)
  
- `eval_GroupLPIV.m`: evaluate the performance of the GLP
  
- `ind_LP.m`: individual LP-IV, it can be used also for non-IV case
  - `HAC.m`: HAC standard errors in the individual LP-IV model
  - `ind_LP_noc.m`: same as `ind_LP.m` but without constant term (for FDAH)
  
- `lag.m`: compute lags of variables

- `panel_LP.m`: panel LP-IV, also works for non-IV case

- `params.m`: parameter values in simulation designs

- `preEmpData.m`: prepare data to be used as input for `GLP.m`

- `resid.m`: Compute whether the iteration converges

## For your own application

You need to:

1. Prepare your data (balanced panel) in long format

2. Specify your model (see below an example):

```matlab
FE = 1;                % unit fixed effects - 1; random effects - 0
par.y_idx = 1;         % dependent variable
par.x_idx = 2;         % policy variable (scalar)
par.c_idx = [3 4 5];   % controls
par.z_idx = [6];       % instrument (preEmpData is written for exogenous control; but we can always specify par.zx_idx and par.zc_idx that instrument z and c separately)
par.nylag = 4;         % specify the number of lags, for y x w and z
par.nxlag = 4;
par.nclag = 4;
par.nzlag = 4;
par.horizon = 24;      % horizons
par.nwtrunc = 25;      % truncation order for Newey-West standard erors (for individual LP-IV)
par.start = '1975-01-01';
par.end   = '2007-12-01';
reg = preEmpData(data, par); % now you have your reg (struct variable) 
```

1. Supply the data struct to `GLP.m`. Before that, it is recommended to run individual LP-IV (to get initial guesses), you can otherwise specify your own guesses (and weight matrix).

```matlab
indOut = ind_LP(reg);  % individual LP-IV

Gmax   = 8;             % maximal number of groups
nInit  = 100;           % number of initializations
bInit  = indOut.b;      % this is the output from individual LP-IV, from which we can draw initial guesses
weight = indOut.asymV;  % this is the weight matrix (I show here the case with L=K)
inference = 1;          % large T inference, with mixed weighting scheme (See SM S1.3) 
[Gr_EST, GIRF, GSE, OBJ, IC] = GLP(tmp, Gmax, nInit, bInit, weight, FE, inference);

```


## References
Huang, J. (2021). Group Local Projections.

McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. Journal of Business & Economic Statistics, 34(4), 574-589.

Mertens, K., & Montiel Olea, J. L. (2018). Marginal tax rates and income: New time series evidence. The Quarterly Journal of Economics, 133(4), 1803-1884.

Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. The Review of Economics and Statistics, 92(4), 1004-1016.
