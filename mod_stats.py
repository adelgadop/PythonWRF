"""
Modified by Alejandro Delgado
Original script from Mario Gavidia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy

def aq_stats(data,polls):
    '''
    Statistics for model evaluation for air quality parameters, according to Emery et al. (2017). 
    Parameters
    ----------
    data: pandas DataFrame
        data with observation (measurements) and model results, where parameters
        have suffixes as '_obs' and '_mod'.
    polls: list str
        name of variables.
        
    Returns
    -------
    stats : pandas DataFrame
        Contain global statistics: Mean bias, Mean Gross Error,
        Root Mean Square Error, IOA, Normalized Mean Bias, Normalized Mean Error,
        Correlation coefficient using numpy, Standard deviation. Not for wind direction
        
    '''
    n    = {}  # number of observations
    MB   = {}  # Mean Bias
    MAGE = {}  # Mean Gross Error
    RMSE = {}  # Root Mean Square Error
    NMB  = {}  # Normalized Mean Bias
    NME  = {}  # Normalized Mean Error
    IOA  = {}  # Index Of Agreement (Willmontt, 1982)
    r    = {}  # Correlation Coefficient based on Numpy module
    Mm   = {}  # Mean of modeling results
    Om   = {}  # Mean of measurements
    Msd  = {}  # Standard deviation of modelling
    Osd  = {}  # Standard deviation of observations

    for pol in polls:
        df  = data[[pol+'_mod',pol+'_obs']].dropna()
        mod = df[pol+'_mod']
        obs = df[pol+'_obs']
        n[pol]    = obs.count()
        MB[pol]   = mod.mean()-obs.mean()
        MAGE[pol] = (mod-obs).abs().mean()
        RMSE[pol] = (((mod - obs)**2).mean())**0.5
        NMB[pol]  = (mod - obs).sum() / obs.sum() * 100
        NME[pol]  = (mod - obs).abs().sum() / obs.sum() * 100
        A = ((mod - obs)**2).sum()
        B = (((mod - obs.mean()).abs() + (obs - obs.mean()).abs())**2).sum()
        IOA[pol]  = 1 - (A / B)
        r[pol]    = np.corrcoef(mod, obs)[0,1]
        Mm[pol]   = mod.mean()
        Om[pol]   = obs.mean()
        Msd[pol]  = mod.std()
        Osd[pol]  = obs.std()
    res = pd.DataFrame({'n':n,'MB':MB,'MAGE':MAGE,'RMSE':RMSE,'NMB':NMB,'NME':NME,'IOA':IOA,'r':r,'Mm':Mm,
                        'Om':Om,'Msd':Msd,'Osd':Osd})
    return res

def wind_dir_diff(Mi, Oi):
    '''
    Difference between Wind directions based in its
    periodic property. Based on Reboredo et al. 2015
    Parameters
    ----------
    Mi : np.float
        Model wind direction.
    Oi : TYPE
        Observed wind direction.
    Returns
    -------
    Wind difference.
    '''
    wd_dif = Mi - Oi
    if Mi < Oi:
        if (np.abs(wd_dif) < np.abs(360 + wd_dif)):
            ans = wd_dif
        elif (np.abs(wd_dif) > np.abs(360 + wd_dif)):
            ans = 360 + wd_dif
    elif Mi > Oi:
        if (np.abs(wd_dif) < np.abs(wd_dif - 360)):
            ans = wd_dif
        elif (np.abs(wd_dif) > np.abs(wd_dif -360)):
            ans = wd_dif - 360
    else:
        ans = 0.0
    
    return(ans)
    
def wind_dir_mb(model_df, obs_df, wd_name='wd'):
    '''
    Calculates wind direction mean bias based in 
    Reboredo et al. 2015
    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observations.
    wd_name : str, optional
        Wind direction column name. The default is 'wd'.
    Returns
    -------
    wd_mb : numpy.float64
        wind direction mean bias.
    '''
    wd_df = pd.DataFrame({
        'mi': model_df,
        'oi': obs_df}) 
    wd_df.dropna(how="any", inplace=True)
    if wd_df.empty:
        wd_mb = np.nan
    else:
        dif = wd_df.apply(lambda row: wind_dir_diff(row['mi'], row['oi']),
                      axis=1)
        wd_mb = dif.mean()    
    return wd_mb


def wind_dir_mage(model_df, obs_df, wd_name='wd'):
    '''
    Calculate wind direction mean absolute error
    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataaFrame with observations.
    wd_name : str, optional
        wind direction column name. The default is 'wd'.
    Returns
    -------
    mage : numpy.float64
        wind direction mean gross error.
    '''
    wd_df = pd.DataFrame({
        'mi': model_df,
        'oi': obs_df})
    wd_df.dropna(how="any", inplace=True)
    if wd_df.empty:
        mage = np.nan
    else:
        dif = wd_df.apply(lambda row: wind_dir_diff(row['mi'], row['oi']),
                      axis=1)
        mage = dif.abs().mean()
    return mage


def met_stats(data,mets):
    '''
    Model performance evaluation for meteorological parameters,
    according to Emery (2001), Reboredo et al (2015), Monk et al (2019).

    Parameters
    ---------
    data: Pandas DataFrame where meteorological parameters are and have suffix '_obs'.
    mets: str list
          meteorological parametes (tc, rh, ws, wd)

    Return
    ------
    Table with statistical results for model performance evaluation as pandas DataFrame.
    Mean Bias (MB), Mean Absolute Gross Error (MAGE), Root Mean Square Error (RMSE), and IOA.
    '''
    n    = {}
    MAGE = {}
    RMSE = {}
    MB   = {}
    IOA  = {}
    r    = {}
    Mm   = {}
    Om   = {}
    Msd  = {}
    Osd  = {}

    for met in mets:
        df        = data[[met+'_mod',met+'_obs']].dropna()
        mod       = df[met+'_mod']
        obs       = df[met+'_obs']
        if met == 'wd':
            n[met]    = obs.count()
            MB[met]   = wind_dir_mb(mod,obs)
            MAGE[met] = wind_dir_mage(mod,obs)
        else:
            n[met]    = obs.count()
            MAGE[met] = (mod - obs).abs().mean()
            RMSE[met] = (((mod - obs)**2).mean())**0.5
            MB[met]   = mod.mean() - obs.mean()
            A = ((mod - obs)**2).sum()
            B = (((mod - obs.mean()).abs() + (obs - obs.mean()).abs())**2).sum()
            IOA[met]  = 1 - (A / B)
            r[met]    = np.corrcoef(mod, obs)[0,1]
            Mm[met]   = mod.mean()
            Om[met]   = obs.mean()
            Msd[met]  = mod.std()
            Osd[met]  = obs.std()
    res = pd.DataFrame({'n':n,'MB':MB,'MAGE':MAGE,'RMSE':RMSE,'IOA':IOA,'r':r,'Mm':Mm,
                        'Om':Om,'Msd':Msd,'Osd':Osd})
    return res

def r_pearson_sig(n, r, alpha, deg_free = 2):
    '''
    Calculate Pearson's R significance. With a two-tail test (non-directional).
    Based on:
    https://medium.com/@shandou/how-to-compute-confidence-interval-for-pearsons-r-a-brief-guide-951445b9cb2d

    Parameters
    ----------
    n : int
        sample size.
    r : float
        Pearson R.
    alpha : float
        test significant level.
    deg_free : int, optional
        degrees of freedom. The default is 2.

    Returns
    -------
    t_cal : float
        Calculated t value.
    t_cri : float
        Critical t value.
    '''
    t_cri = scipy.stats.t.ppf(1 - alpha / 2.0, deg_free)
    t_cal = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    if t_cal > t_cri:
        print("Significant linear relationship")
    else:
        print("No significant linear relationship")
    return (t_cal, t_cri)

def r_pearson_confidence_interval(n, r, alpha):
    '''
    Calculate Pearson's R confidence intervals, using two-tail test.
    Based on:
    http://onlinestatbook.com/2/estimation/correlation_ci.html
    https://medium.com/@shandou/how-to-compute-confidence-interval-for-pearsons-r-a-brief-guide-951445b9cb2d

    Parameters
    ----------
    n : int
        sample size.
    r : float
        Pearson's R.
    alpha : float
        confidence level (e.g. if 95% then alpha = 0.05).

    Returns
    -------
    r_lower : float
        lower CI.
    r_upper : float
        upper CI.
    '''
    alph = 0.05 / 2.0 # two-tail test:
    z_critical = scipy.stats.norm.ppf(1 - alph)
    # r to z' by Fisher's z' transform:
    z_prime =0.5 * np.log((1 + r) / (1 - r))
    # Sample standard error:
    se = 1 / np.sqrt(n - 3)
    # Computing CI using z':
    ci_lower = z_prime - z_critical * se
    ci_upper = z_prime + z_critical * se
    # Converting z' back to r values:
    r_lower = np.tanh(ci_lower)
    r_upper = np.tanh(ci_upper)
    return (r_lower, r_upper)





