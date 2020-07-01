# GLP

This repository contains the code and data for “Group Local Projection”. Feel free to contact me ([jiaming.huang@barcelonagse.eu](jiaming.huang@barcelonagse.eu) if you have any doubts on the codes.

## Replication

For the simulation study, the results can be replicated using `SIM_KnownGroupNum.m` (Section 6.2), `SIM_UnknownGroupNum.m` (Section 6.3) and `SIM_NoTrueGroup.m` (Section 6.4).

For the empirical application, `EMP_Main.m` gives the baseline results (individual LP-IV, panel LP-IV and the GLP). `EMP_FAVAR.m` shows the FAVAR results. 

The FAVAR application involves two additional folders: 

1. fred: used to estimate the factor model, which is borrowed from the FRED-MD package by McCracken and Ng (2016). I modify the `factors_em.m` file to include the Onatski (2010) criterion. 

2. svar: used to estimate the SVAR-IV model after factor estimation. Here I use the proxy-SVAR upackage provided by Mertens and Montiel-Olea (2018). Alternative inference methods are included (wild bootstrap, delta method and Jentsch Lunsford MBB)

Auxiliary functions in the routine folder are:

- `DGP.m` and `DGP_no_group.m`: generate simulated data

- `GroupLPIV.m`: general function for the GLP estimation
  - `GroupLPIV_TrueGroup.m`: infeasible GLP in which the latent group is known
  - `GroupLPIV_TrueIRF.m`: in simulations where the true IRs are known, I use them as the only initial guess to speed up the simulation
  
- `eval_GroupLPIV.m`: evaluate the performance of the GLP
  - `eval_GroupLPIV_noTrue.m`: the same function, for the case where there is no group structure
  
- `ind_LP.m`: individual LP-IV, it can be used also for non-IV case
  - `ind_HAC.m`: HAC standard errors in the individual LP-IV model
  
- `lag.m`: compute lags of variables

- `panel_LP.m`: panel LP-IV, also works for non-IV case

- `params.m`: parameter values in simulation designs

- `prepare_GLP.m`: prepare data to be used as input for `GroupLPIV.m`

- `resid.m`: Compute whether the iteration converges

- `transx.m`: transform variables (comparable to `transxf.m` in the factor estimation package)

## For your own application

All you need is:

1. Prepare your data (__balanced panel__) in long format

2. Specify your model (see below an example):

```matlab
par.y_idx = 1;         % dependent variable
par.x_idx = 2;         % policy variable (scalar)
par.w_idx = [3 4 5];   % external controls
par.z_idx = [6];       % instrument (now the code is written only for scalar instrument, modification is easy)
par.nylag = 4;         % specify the number of lags
par.nxlag = 4;
par.nwlag = 4;
par.nzlag = 4;
par.horizon = 24;      % horizons
par.start = '1975-01-01';  % time range, the date format depends on the par.date in your data
par.end   = '2007-12-01';
reg = prepare_GLP(data, par); % now you have your reg (struct variable) 
```

3. Supply the data struct to `GroupLPIV.m`. You can specify

- `K`: the number of groups
- `ninit`: number of initial guesses
- `tsls`: 1 if two-stage least squares; 0 if IV estimator
- `FE`: 1 if consider unit fixed effects; 0 if not
- `binit`: a pool of initial guesses, e.g. the individual LP-IV estimates

```matlab
[Group, GIRF, GSE, Qpath, gpath, bpath] = GroupLPIV(reg, K, ninit, tsls, FE, binit)
```


## References

McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. Journal of Business & Economic Statistics, 34(4), 574-589.

Mertens, K., & Montiel Olea, J. L. (2018). Marginal tax rates and income: New time series evidence. The Quarterly Journal of Economics, 133(4), 1803-1884.

Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. The Review of Economics and Statistics, 92(4), 1004-1016.
