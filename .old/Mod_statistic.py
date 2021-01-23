"""
Modified by Alejandro Delgado
Original script from Mario Gavidia
Source from Emery et al. (2017)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def complete_cases(model_df, obs_df, var):
    '''
    Create a dataframe with complete cases rows
    for model evaluation for an evaluated variable

    Parameters
    ----------
    model_df : pandas DataFrame
        model results.
    obs_df : pandas DataFrame
        observations .
    var : str
        variable column name.

    Returns
    -------
    df : pandas DataFrame
        dataframe with no NaN values.

    '''
    data = pd.merge(obs_df, model_df,
                on=['local_date','station','code'],
                suffixes=('_obs', '_mod'))
    df = pd.concat([data[var+'_mod'], data[var+'_obs']],
                   axis=1,
                   keys=["wrf", "obs"])
    df.dropna(how="any", inplace=True)
    return df


def mean_bias(model_df, obs_df, var):
    '''
    Calculate Mean Bias

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.

    Returns
    -------
    mb : numpy.float64
        mean bias.

    '''
    df = complete_cases(model_df, obs_df, var)
    mb = df.wrf.mean() - df.obs.mean()
    return mb

def mean_gross_error(model_df, obs_df, var):
    '''
    Calcualte mean gross error

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.

    Returns
    -------
    me : numpy.float64
        Mean gross error.

    '''
    df = complete_cases(model_df, obs_df, var)
    me = (df.wrf - df.obs).abs().mean()
    return me

def root_mean_square_error(model_df, obs_df, var):
    '''
    Calcualte Root mean square error

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.

    Returns
    -------
    rmse : numpy.float64
        root mean square error.

    '''
    df = complete_cases(model_df, obs_df, var)
    rmse = (((df.wrf - df.obs)**2).mean())**0.5
    return rmse


def normalized_mean_bias(model_df, obs_df, var):
    '''
    Calculate the normalized mean bais

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.

    Returns
    -------
    nmb : numpy.float64
        normalized mean bias.

    '''
    if obs_df[var].dropna().empty:
        nmb = np.nan
    else:
        df = complete_cases(model_df, obs_df, var)
        nmb = (df.wrf - df.obs).sum() / df.obs.sum() * 100
    return nmb

def normalized_mean_error(model_df, obs_df, var):
    '''
    Calculate normalized mean error

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.

    Returns
    -------
    nme : numpy.float64
        normalized mean error.

    '''
    if obs_df[var].dropna().empty:
        nme = np.nan
    else:
        df = complete_cases(model_df, obs_df, var)
        nme = ((df.wrf - df.obs).abs().sum() /
               df.obs.sum() * 100)
    return nme

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
    data = pd.merge(obs_df, model_df, on=['local_date','station','code'],
            suffixes=('_obs', '_mod'))
    wd_df = pd.DataFrame({
        'mi': data[wd_name+'_mod'].values,
        'oi': data[wd_name+'_obs'].values})
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
    data = pd.merge(obs_df, model_df, on=['local_date','station','code'],
            suffixes=('_obs', '_mod'))
    wd_df = pd.DataFrame({
        'mi': data[wd_name+'_mod'].values,
        'oi': data[wd_name+'_obs'].values})
    wd_df.dropna(how="any", inplace=True)
    if wd_df.empty:
        mage = np.nan
    else:
        dif = wd_df.apply(lambda row: wind_dir_diff(row['mi'], row['oi']),
                      axis=1)
        mage = dif.abs().mean()
    return mage


def all_stats(model_df, obs_df, var, to_df=False):
    '''
    Calculates recommended statistics from Emery et al. (2017)

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observation.
    var : str
        Name of variable.
    to_df : Bool, optional
        Ouput in DataFrame. The default is False.

    Returns
    -------
    results : dict or DataFrame
        MB, RMSE, NMB, NME, R, Model and Obs means and std.

    '''
    data = pd.merge(obs_df, model_df, on=['local_date','station','code'],
            suffixes=('_obs', '_mod'))
    if var == 'wd':
        data = pd.merge(obs_df, model_df, on=['local_date','station','code'],
                suffixes=('_obs', '_mod'))
        results = {
            'MB': wind_dir_mb(model_df, obs_df),
            'ME': wind_dir_mage(model_df, obs_df),
            'aqs': data['station'].unique()[0]}
    else:
        results = {
            'MB': mean_bias(model_df, obs_df, var),
            'ME': mean_gross_error(model_df, obs_df, var),
            'RMSE': root_mean_square_error(model_df, obs_df, var),
            'NMB': normalized_mean_bias(model_df, obs_df, var),
            'NME': normalized_mean_error(model_df, obs_df, var),
            'R': data[var+'_mod'].corr(data[var+'_obs']),
            'Om': data[var+'_obs'].mean(),
            'Mm': data[var+'_mod'].mean(),
            'Ostd': data[var+'_obs'].std(),
            'Mstd': data[var+'_mod'].std(),
            'aqs': data['station'].unique()[0]}

    if to_df:
        results = pd.DataFrame(results, index=[var])
    return results

def all_var_stats_per_station(model_df, obs_df,para=['o3','co','nox','tc','ws','rh'], to_df=False):
    '''
    Calculate all stats for each observation parameter

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observations.
    to_df : Bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    results : dict or DataFrame
        All Statistic for all observation variables.

    '''
    data = pd.merge(obs_df, model_df, on=['local_date','station','code'],
            suffixes=('_obs', '_mod'))
    var_to_eval = para
    results = {}
    for k in  data['station'].unique():
        for var in var_to_eval:
            results[var] = all_stats(model_df, obs_df, var)

    if to_df:
        results = pd.DataFrame.from_dict(results,
                                         orient='index')
    return results

def some_vars_stats_per_station(model_df, obs_df, var, to_df=False):
    '''
    Calculate all stats for each observation parameter

    Parameters
    ----------
    model_df : pandas DataFrame
        DataFrame with model output.
    obs_df : pandas DataFrame
        DataFrame with observations.
    to_df : Bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    results : dict or DataFrame
        All Statistic for all observation variables.

    '''
    var_to_eval = var
    results = {}
    for var in var_to_eval:
        results[var] = all_stats(model_df, obs_df, var)

    if to_df:
        results = pd.DataFrame.from_dict(results, 
                                         orient='index')        
    return results



def all_aqs_all_vars(model_dic, obs_dic, to_df=True, sort_pol = False, csv = False):
    '''
    Calculate all statistic for all variables for all
    evaluated stations

    Parameters
    ----------
    model_dic : dict
        Dictionaryy containing data frames with station data from model.
    obs_dic : dict
        Dictionaryy containing data frames with station data from aqs.
    to_df : bool, optional
        Return a data frame. The default is True.
    sort_pol : bool, optional
        when to_df=True output sorted by pol. The default is False.
    csv : bool, optional
        When to_df=Truem export it to csv. The default is False.

    Returns
    -------
    result : pandas DataFrame or dict
        All statistic for all variaables for all aqs.

    '''
    result = {}
    for k in model_dic:
        result[k] = all_var_stats_per_station(model_dic[k], obs_dic[k],
                                              to_df=to_df)
    if to_df:
        result = pd.concat(result.values())
        if sort_pol:
            result.sort_index(inplace=True)
        if csv:
            file_name = '_'.join(result.index.unique().values) + "_stats.csv"
            result.to_csv(file_name, sep=",", index_label="pol")       
        
    return result

def all_aqs_some_vars(model_dic, obs_dic, var, to_df=True, 
                     sort_pol = False, csv = False):
    '''
    Calculate all statistic for all variables for all
    evaluated stations

    Parameters
    ----------
    model_dic : dict
        Dictionaryy containing data frames with station data from model.
    obs_dic : dict
        Dictionaryy containing data frames with station data from aqs.
    to_df : bool, optional
        Return a data frame. The default is True.
    sort_pol : bool, optional
        when to_df=True output sorted by pol. The default is False.
    csv : bool, optional
        When to_df=Truem export it to csv. The default is False.

    Returns
    -------
    result : pandas DataFrame or dict
        All statistic for all variaables for all aqs.

    '''
    result = {}
    for k in model_dic:
        result[k] = some_vars_stats_per_station(model_dic[k], obs_dic[k], var,to_df=to_df)
    if to_df:
        result = pd.concat(result.values())
        if sort_pol:
            result.sort_index(inplace=True)
        if csv:
            file_name = '_'.join(result.index.unique().values) + "_stats.csv"
            result.to_csv(file_name, sep=",", index_label="pol")       
        
    return result

def global_stat(model_dic, obs_dic, csv=False):
    '''
    Calculates the global statistics  

    Parameters
    ----------
    model_dic : dict
        Dictionary containing data frames with station data from model.
    obs_dic : dict
        Dictionary containing data frames with station data from aqs.
    csv : bool, optional
        Export the value as csv. The default is False.

    Returns
    -------
    stats : pandas DataFrame
        Contain global statistics.

    '''
    model_df = pd.concat(model_dic)
    obs_df =pd.concat(obs_dic)
    
    stats = all_var_stats_per_station(model_df, obs_df, to_df=True)
    stats.drop(labels='aqs', axis=1, inplace=True)
    if csv:
        file_name = '_'.join(stats.index.values) + "_global_stats.csv"
        stats.to_csv(file_name, sep=",", index_label='pol')
    return stats

    
def global_stat_some_vars(model_dic, obs_dic, var, csv=False):
    '''
    Calculates the global statistics  

    Parameters
    ----------
    model_dic : dict
        Dictionary containing data frames with station data from model.
    obs_dic : dict
        Dictionary containing data frames with station data from aqs.
    csv : bool, optional
        Export the value as csv. The default is False.

    Returns
    -------
    stats : pandas DataFrame
        Contain global statistics.

    '''
    model_df = pd.concat(model_dic)
    obs_df =pd.concat(obs_dic)
    
    stats = some_vars_stats_per_station(model_df, obs_df,var, to_df=True)
    stats.drop(labels='aqs', axis=1, inplace=True)
    if csv:
        file_name = '_'.join(stats.index.values) + "_global_stats.csv"
        stats.to_csv(file_name, sep=",", index_label='pol')
    return stats

def mod_Stats(data,polls=['o3','no','no2','co','tc','rh']):
    '''
    Statistics for model evaluation, according to Emery et al. (2017).
    
    Parameters
    ----------
    data: pandas DataFrame
        data with observation (measurements) and model results, where parameters
        have suffixes as '_obs' and '_mod'.
    polls: str
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
        MB.append(df[pol+'_obs'].mean()-df[pol+'_mod'].mean())
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
    Ostd = pd.DataFrame([dict(zip(polls,Ostd))]).T.rename(columns={0:'Obs SD'})
    Mstd = pd.DataFrame([dict(zip(polls,Mstd))]).T.rename(columns={0:'Mod SD'})
    ''' 
    Join
    '''
    stats = pd.concat([n,MB,MGE,RMSE,NMB,NME,r, Ostd, Mstd],axis=1)
    return stats 
