"""
Modified by Alejandro Delgado
Original script from Mario Gavidia
Source from Emery et al. (2017)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mod_Stats(data,polls):
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
        Root Mean Square Error, Normalized Mean Bias, Normalized Mean Error,
        Correlation coefficient using numpy, Standard deviation. Not for wind direction
        
    '''
    '''
    Mean bias
    '''
    MB = []
    n = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        n.append(df[pol+'_obs'].count())
        MB.append(df[pol+'_mod'].mean()-df[pol+'_obs'].mean())
    MB = pd.DataFrame([dict(zip(polls,MB))]).T.rename(columns={0:'MB'})
    n  = pd.DataFrame([dict(zip(polls,n))]).T.rename(columns={0:'n'})
    
    '''
    Mean Gross Error
    '''
    MGE = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        MGE.append((df[pol+'_mod'] - df[pol+'_obs']).abs().mean())
    MGE = pd.DataFrame([dict(zip(polls,MGE))]).T.rename(columns={0:'MGE'})
    '''
    Root Mean Square Error
    '''
    RMSE = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        RMSE.append((((df[pol+'_mod'] - df[pol+'_obs'])**2).mean())**0.5)
    RMSE = pd.DataFrame([dict(zip(polls,RMSE))]).T.rename(columns={0:'RMSE'})
    
    '''
    Normalized Mean Bias
    '''
    NMB = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        NMB.append( (df[pol+'_mod'] - df[pol+'_obs']).sum() / df[pol+'_obs'].sum() * 100 )
    NMB = pd.DataFrame([dict(zip(polls,NMB))]).T.rename(columns={0:'NMB'})
    '''
    Normalized Mean Error
    '''
    NME = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        NME.append( ((df[pol+'_mod'] - df[pol+'_obs']).abs().sum() /
                     df[pol+'_obs'].sum() * 100) )
    NME = pd.DataFrame([dict(zip(polls,NME))]).T.rename(columns={0:'NME'})
   
    '''
    Index of Agreement (Willmontt, 1982)
    '''
    IOA = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        A = ((df[pol+'_mod']-df[pol+'_obs'])**2).sum()
        B = (((df[pol+'_mod']-df[pol+'_obs'].mean()).abs() + \
            (df[pol+'_obs']-df[pol+'_obs'].mean()).abs())**2).sum()
        IOA.append(1-(A/B))
    IOA = pd.DataFrame([dict(zip(polls,IOA))]).T.rename(columns={0:'IOA'})

    '''
    Correlation coefficient using numpy
    '''
    r = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        r.append(np.corrcoef(df[pol+'_mod'], df[pol+'_obs'])[0,1])
    r = pd.DataFrame([dict(zip(polls,r))]).T.rename(columns={0:'r'})
    '''
    Standard deviation
    '''
    Ostd = []
    Mstd = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        Ostd.append(df[pol+'_obs'].std())
        Mstd.append(df[pol+'_mod'].std())
    Ostd = pd.DataFrame([dict(zip(polls,Ostd))]).T.rename(columns={0:'Osd'})
    Mstd = pd.DataFrame([dict(zip(polls,Mstd))]).T.rename(columns={0:'Msd'})
    '''
    Observed mean
    '''
    Om = []
    Mm = []
    for pol in polls:
        df = data[[pol+'_obs',pol+'_mod']].dropna()
        Om.append(df[pol+'_obs'].mean())
        Mm.append(df[pol+'_mod'].mean())
    Om = pd.DataFrame([dict(zip(polls,Om))]).T.rename(columns={0:'Om'})
    Mm = pd.DataFrame([dict(zip(polls,Mm))]).T.rename(columns={0:'Mm'})

    '''
    Join
    '''
    stats = pd.concat([n,MB,MGE,RMSE,NMB,NME,IOA,r,Mm,Om, Mstd, Ostd],axis=1)
    return stats

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
        'mi': model_df[wd_name+'_mod'].values,
        'oi': obs_df[wd_name+'_obs'].values}) 
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
        'mi': model_df[wd_name+'_mod'].values,
        'oi': obs_df[wd_name+'_obs'].values})
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
    Model performance evaluation for meteorological parameters, according to Emery (2001), Reboredo et al (2015), Monk et al (2019).

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







