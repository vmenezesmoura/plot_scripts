import pandas as pd

def fig_1x4_4subplot(s0: pd.Series, s: pd.Series, xlabel_subplot0: str, suptitle: str, figsize: tuple[int] = (8, 4), width_ratios: list[float] = [2, 1.5, 1, 1]):
    '''
    Plots a 1x4 grid of subplots comparing two series (observed vs. predicted) with various statistical representations.

    Parameters:
    -----------
    s0 : pd.Series
        The observed data series.
    s : pd.Series
        The predicted data series.
    xlabel_subplot0 : str
        Label for the x-axis of the first subplot.
    suptitle : str
        The main title for the figure.
    figsize : tuple[int], optional
        Size of the figure (width, height), default is (8, 4).
    width_ratios : list[float], optional
        Width ratios for the subplots, default is [2, 1.5, 1, 1].

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The resulting figure object.
    df_s : pd.DataFrame
        A DataFrame containing the observed and predicted data.

    Description:
    ------------
    This function creates a figure with 4 subplots arranged in a 1x4 grid:
    1. Histogram comparison of the observed vs. predicted data with statistics (mean, median).
    2. Scatter plot comparing observed vs. predicted values along with a linear fit and R² value.
    3. Histogram of the prediction errors, showing the median error and normalized BIAS.
    4. Boxplots showing the distribution of observed and predicted values.

    Additional statistics (mean, range, standard deviation, interquartile range, skewness, and kurtosis) are calculated for each series and displayed as text or lines in the respective plots.

    The function also performs linear regression to compute the slope and intercept for the scatter plot comparison.

    Colors, titles, and labels are configured for each subplot. The layout is tightened for better visualization, and the overall figure title is set to the provided 'suptitle'.
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    # Set up the figure grid and color configurations
    fig_grade = (1, 4)
    color_data = ['red', 'blue']  # Colors for observed and predicted data
    color_comparison = 'purple'   # Color for comparison elements (scatter plot and error)
    color_iqr = ['mistyrose', 'lavender']  # Colors for interquartile range areas
    color_linestat = {
        'mean': ['darkred', 'darkblue'],  # Colors for mean lines
        'median': ['black', 'white'],     # Colors for median lines
        'R2': 'white',                    # Color for R² line in the scatter plot
        'nbias': 'white'                  # Color for normalized BIAS line in error plot
    }
    
    # Axis labels for the subplots
    ylabel_subplot0 = 'densidade normalizada'
    xlabel_subplot1 = 'obsevado'
    ylabel_subplot1 = 'previsto'
    xlabel_subplot2 = 'erro'
    ylabel_subplot2 = ylabel_subplot0
    xlabel_subplot3 = ''
    ylabel_subplot3 = xlabel_subplot0
    
    # Create lists of x and y labels for the subplots using the defined label variables
    xlabel = [locals()[f'xlabel_subplot{i}'] for i in range(fig_grade[1])]
    ylabel = [locals()[f'ylabel_subplot{i}'] for i in range(fig_grade[1])]
    
    # List of statistical measures to calculate
    list_stats = ['mean', 'range', 'std', 'iqr', 'skew', 'kurtosis']

    # Prepare the data by finding the common index and concatenating the series into a DataFrame
    common_index = s0.index.intersection(s.index)
    df_s = pd.concat([s0, s], index=1)
    max_value = max(s0.max(), s.max())
    min_value = min(s0.min(), s.min())
    max_abs_error = (s - s0).abs().max
    
    # Define histogram bins for data and error distributions
    bins_minamax = range(min_value, max_value * 1.1, (max_value - min_value) / 30)
    bins_absmaxamax = range(-max_abs_error, max_abs_error * 1.1, max_abs_error / 15)
    
    # Perform linear regression to find the slope and intercept of the line fitting the observed vs. predicted data
    slope, intercept = np.polyfit(s0, s, 1)

    # Calculate statistical measures for observed and predicted data
    for k in ['0', '']:
        locals()[f's{k}_'] = locals()[f's{k}']
        locals()[f's{k}'] = locals()[f's{k}_'].loc[common_index]
        locals()[f'{list_stats[0]}_s{k}'] = locals()[f's{k}'].mean()
        locals()[f'{list_stats[1]}_s{k}'] = locals()[f's{k}'].max() - locals()[f's{k}'].min()
        locals()[f'{list_stats[2]}_s{k}'] = locals()[f's{k}'].std()
        locals()[f'{list_stats[3]}_s{k}'] = locals()[f's{k}'].quantile(0.75) - locals()[f's{k}'].quantile(0.25)
        locals()[f'{list_stats[4]}_s{k}'] = locals()[f's{k}'].skew()
        locals()[f'{list_stats[5]}_s{k}'] = locals()[f's{k}'].kurtosis()
    
    # Create Series to store calculated statistics for both observed and predicted data
    stats_s0 = pd.Series(
        [
        locals()[f'{list_stats[0]}_s0'],
        locals()[f'{list_stats[1]}_s0'],
        locals()[f'{list_stats[2]}_s0'],
        locals()[f'{list_stats[3]}_s0'],
        locals()[f'{list_stats[4]}_s0'],
        locals()[f'{list_stats[5]}_s0']
        ], 
        index=list_stats, 
        name=xlabel_subplot1
    )    
    stats_s = pd.Series(
        [
        locals()[f'{list_stats[0]}_s'],
        locals()[f'{list_stats[1]}_s'],
        locals()[f'{list_stats[2]}_s'],
        locals()[f'{list_stats[3]}_s'],
        locals()[f'{list_stats[4]}_s'],
        locals()[f'{list_stats[5]}_s']
        ], 
        index=list_stats, 
        name=ylabel_subplot1
    )
    
    # Concatenate statistics into a DataFrame for display
    df_stats = pd.concat([stats_s0, stats_s], index=1)

    # Calculate correlation, R², normalized RMSE, and normalized BIAS between observed and predicted data
    corr = df_s.corr().iloc[0,1]
    r2 = r2_score(s, LinearRegression().fit(df_s[s0.name], s).predict(df_s[s0.name]))
    nrmse =   np.sqrt(mean_squared_error(s0, s)) / locals()['range_s0'] if locals()['range_s0'] != 0 else np.nan
    nbias =   (s - s0).mean() / locals()['range_s0'] if locals()['range_s0'] != 0 else np.nan

    # Create the figure and subplots using GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], width_ratios=width_ratios)
    axes = []

    for i in range(fig_grade[1]):
        axes.append(fig.add_subplot(gs[i]))

    # Configure each subplot with appropriate titles and labels
    axes[0].set_title(f'Distribuição {xlabel_subplot1} vs {ylabel_subplot1}')
    axes[0].set_xlabel(xlabel_subplot0)
    axes[0].set_ylabel(ylabel_subplot0)

    axes[1].set_title(f'Correlação')
    axes[1].set_xlabel(xlabel_subplot1)
    axes[1].set_ylabel(ylabel_subplot1)

    axes[2].set_title(f'Erro')
    axes[2].set_xlabel(xlabel_subplot2)
    axes[2].set_ylabel(ylabel_subplot2)

    axes[3].set_title(f'Boxplot')
    axes[3].set_xlabel(xlabel_subplot3)
    axes[3].set_ylabel(ylabel_subplot3)

    # Plot data on the first subplot: histograms with mean and median lines and IQR areas
    axes[0].axvspan(s0.quantile(0.25), s0.quantile(0.75), color=color_iqr[0], label=f'IQR {xlabel_subplot1}', alpha=0.3)
    axes[0].axvspan(s.quantile(0.25), s.quantile(0.75), color=color_iqr[1], label=f'IQR {ylabel_subplot1}', alpha=0.3)
    axes[0].axvline(x=s0.median(), color=color_linestat['median'][0], ls='--', lw=1, label=f'Mediana {xlabel_subplot1}')
    axes[0].axvline(x=s.median(), color=color_linestat['median'][1], ls='--', lw=1, label=f'Mediana {ylabel_subplot1}')
    axes[0].axvline(x=s0.mean(), color=color_linestat['mean'][0], lw=1, label=f'Média {xlabel_subplot1}')
    axes[0].axvline(x=s.mean(), color=color_linestat['mean'][1], lw=1, label=f'Média {ylabel_subplot1}')
    df_s.hist(density=True, histtype='bar', bins=bins_minamax, ax=axes[0], color=color_data, label=[xlabel_subplot1, ylabel_subplot1])
    axes[0].legend(loc='best', fontsize='x-small')

    # Plot data on the second subplot: scatter plot with linear fit and R² annotation
    axes[1].scatter(s0, s, s=5, color=color_comparison, label=f'r$^2$={r2:.2f}')
    axes[1].plot(s0, s0 * slope + intercept, lw=1, ls='--', color=color_linestat['R2'], label=f'corr={corr:.2f}, r$^2$={r2:.2f}')
    axes[1].plot([min_value, max_value], [min_value, max_value], lw=1, color='gray', ls=':', label=f'{xlabel_subplot1}={ylabel_subplot1}')
    axes[1].legend(loc='best', fontsize='x-small')

    # Plot data on the third subplot: error histogram with median error and normalized BIAS annotation
    (s - s0).hist(density=True, histtype='bar', bins=bins_absmaxamax, ax=axes[2], color=color_comparison, label=f'{xlabel_subplot1}-{ylabel_subplot1}')
    axes[2].axvline(x=(s - s0).median(), color=color_linestat['median'][0], ls='--', lw=1, label=f'Mediana do erro')
    axes[2].axvline(x=nbias * (locals()['range_s0']), color=color_linestat['nbias'], lw=1, label=f'nBIAS={nbias:.2f}')
    axes[2].legend(loc='best', fontsize='x-small')

    # Plot data on the fourth subplot: boxplots for observed and predicted data
    df_s.plot(kind='box', vert=1, widths=0.5, grid=0, ax=axes[3], color=dict(boxes=color_data, whiskers=color_data, medians=color_linestat['median'], caps=color_data))

    # Add grid and legend to subplots
    for ax in axes:
        ax.grid(axis='both')
        ax.legend()

    # Add main title and adjust layout for better display
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, df_s
