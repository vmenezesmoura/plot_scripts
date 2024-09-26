import pandas as pd

def fig_1x4_4subplot(s0: pd.Series, s: pd.Series, xlabel_subplot0: str, suptitle: str, figsize: tuple[int] = (14, 4), width_ratios: list[float] = [2, 1.5, 1, 1]):
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
    df_stats : pd.DataFrame
        A DataFrame containing the statistics of observed and predicted data.

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
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    # Set up the figure grid and color configurations
    fig_grade = (1, 4)
    color_data = ['red', 'blue']  # Colors for observed and predicted data
    color_comparison = 'purple'   # Color for comparison elements (scatter plot and error)
    color_iqr = ['mistyrose', 'lavender']  # Colors for interquartile range areas
    color_linestat = {
        'mean': ['darkred', 'darkblue'],  # Colors for mean lines
        'median': ['yellow', 'green'],     # Colors for median lines
        'R2': 'lightgray',                    # Color for R² line in the scatter plot
        'nbias': 'lightgray'                  # Color for normalized BIAS line in error plot
    }
    
    # Axis labels for the subplots
    ylabel_subplot0 = 'densidade normalizada'
    xlabel_subplot1 = 'obsevado'
    ylabel_subplot1 = 'previsto'
    xlabel_subplot2 = 'erro'
    ylabel_subplot2 = ylabel_subplot0
    xlabel_subplot3 = ''
    ylabel_subplot3 = xlabel_subplot0
    
    # List of statistical measures to calculate
    list_stats = ['mean', 'range', 'std', 'iqr', 'skew', 'kurtosis']

    # Prepare the data by finding the common index and concatenating the series into a DataFrame
    common_index = s0.index.intersection(s.index)
    max_value = max(s0.max(), s.max())
    min_value = min(s0.min(), s.min())
    s0 = s0.loc[common_index]
    s = s.loc[common_index]
    
    # Define histogram bins for data and error distributions
    bins_minamax = np.arange(min_value, max_value * 1.1, (max_value - min_value) / 30)

    # Calculate statistical measures for observed and predicted data
    for k in ['0', '']:
        locals()[f'{list_stats[0]}_s{k}'] = locals()[f's{k}'].mean()
        locals()[f'{list_stats[1]}_s{k}'] = locals()[f's{k}'].max() - locals()[f's{k}'].min()
        locals()[f'{list_stats[2]}_s{k}'] = locals()[f's{k}'].std()
        locals()[f'{list_stats[3]}_s{k}'] = locals()[f's{k}'].quantile(0.75) - locals()[f's{k}'].quantile(0.25)
        locals()[f'{list_stats[4]}_s{k}'] = locals()[f's{k}'].skew()
        locals()[f'{list_stats[5]}_s{k}'] = locals()[f's{k}'].kurtosis()
    
    # Perform linear regression to find the slope and intercept of the line fitting the observed vs. predicted data
    df_s = pd.concat([s0, s], axis=1)
    max_abs_error = (s - s0).abs().max()
    bins_absmaxamax = np.arange(-max_abs_error, max_abs_error * 1.1, max_abs_error / 15)
    slope, intercept = np.polyfit(s0, s, 1)

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
    df_stats = pd.concat([stats_s0, stats_s], axis=1)

    # Calculate correlation, R², normalized RMSE, and normalized BIAS between observed and predicted data
    corr = df_s.corr().iloc[0,1]
    r2 = r2_score(s0, s)
    nrmse =   np.sqrt(mean_squared_error(s0, s)) / stats_s0['range'] if stats_s0['range'] != 0 else np.nan
    nbias =   (s - s0).mean() / stats_s0['range'] if stats_s0['range'] != 0 else np.nan

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
    axes[0].axvspan(s0.quantile(0.25), s0.quantile(0.75), color=color_iqr[0], label=f'IQR {xlabel_subplot1[:3]}')
    axes[0].axvspan(s.quantile(0.25), s.quantile(0.75), color=color_iqr[1], label=f'IQR {ylabel_subplot1[:4]}')
    axes[0].axvline(x=s0.median(), color=color_linestat['median'][0], ls='-', lw=1, label=f'mediana {xlabel_subplot1[:3]}={s0.median():.2f}')
    axes[0].axvline(x=s.median(), color=color_linestat['median'][1], ls='-', lw=1, label=f'mediana {ylabel_subplot1[:4]}={s.median():.2f}')
    axes[0].hist(s0, density=True, bins=bins_minamax, color=color_data[0], label=xlabel_subplot1, alpha=0.5)
    axes[0].hist(s, density=True, bins=bins_minamax, color=color_data[1], label=ylabel_subplot1, alpha=0.5)
    axes[0].axvline(x=stats_s0['mean'], color=color_linestat['mean'][0], ls=':', label=f'média {xlabel_subplot1[:3]}={stats_s0['mean']:.2f}')
    axes[0].axvline(x=stats_s['mean'], color=color_linestat['mean'][1], ls=':', label=f'média {ylabel_subplot1[:4]}={stats_s['mean']:.2f}')

    # Plot data on the second subplot: scatter plot with linear fit and R² annotation
    axes[1].scatter(s0, s, s=1, color=color_comparison, label='dados cruzados', alpha=0.3)
    axes[1].plot([min_value, max_value], [min_value, max_value], lw=1, color='black', ls=':', label='identidade')
    axes[1].plot(s0, slope * s0 + intercept, lw=1, ls='--', color=color_linestat['R2'], label=f'ajuste R$^2$={r2:.2f}\n[Corr={corr:.2f}]')

    # Plot data on the third subplot: error histogram with median error and normalized BIAS annotation
    axes[2].hist(s - s0, density=True, bins=bins_absmaxamax, color=color_comparison, label='erro da previsão', alpha=0.7)
    axes[2].axvline(x=nbias * stats_s0['range'], color=color_linestat['nbias'], ls='--', lw=1, label=f'nBIAS={nbias:.2f}\n[BIAS={nbias * stats_s0['range']:.2f}]\n[nRMSE={nrmse:.2f}]')

    # Plot data on the fourth subplot: boxplots for observed and predicted data
    box = axes[3].boxplot(df_s, patch_artist=True, widths=0.5, labels=[xlabel_subplot1[:3], ylabel_subplot1[:4]])

    # Aplicando estilos para cada boxplot individualmente
    for m, (patch, color) in enumerate(zip(box['boxes'], color_data)):
        patch.set_facecolor(color_iqr[m])
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)  # Espessura das bordas da caixa
        
        # Aplicando o mesmo estilo para as outras partes do boxplot
        for n in range(m * 2, m * 2 + 2):
            box['whiskers'][n].set_color(color)
            box['whiskers'][n].set_linewidth(1.5)  # Espessura dos whiskers
        
        for n in range(m * 2, m * 2 + 2):
            box['caps'][n].set_color(color_linestat['mean'][m])
            box['caps'][n].set_linewidth(1.5)  # Espessura dos caps

        # Aplicando o estilo na linha da mediana com maior espessura
        box['medians'][m].set_color(color_linestat['median'][m])
        box['medians'][m].set_linewidth(3)  # Espessura maior para a mediana
        
        # Estilo dos outliers
        for flier in box['fliers']:
            flier.set(marker='.', color=color_linestat['mean'][m], markersize=4, alpha=0.7)

    # Add grid and legend to subplots
    for i in range(len(axes)):
        axes[i].grid(axis='both')
        if i%fig_grade[1] != fig_grade[1] - 1:
            axes[i].legend(fontsize=7)

    # Add main title and adjust layout for better display
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, df_stats
