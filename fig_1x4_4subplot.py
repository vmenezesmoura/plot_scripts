import pandas as pd

def fig_1x4_4subplot(s0: pd.Series, s: pd.Series, xlabel_subplot0:str, suptitle: str, figsize: tuple[int] = (8, 4), width_ratios: list[float] = [2, 1.5, 1, 1]):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    fig_grade = (1, 4)
    color_data = ['red', 'blue']
    color_comparison = 'purple'
    color_iqr = ['mistyrose', 'lavender']
    color_linestat = {
        'mean':['darkred', 'darkblue'],
        'median':['black', 'white'],
        'R2':'white',
        'nbias':'white'
    }
    ylabel_subplot0 = 'densidade normalizada'
    xlabel_subplot1 = 'obsevado'
    ylabel_subplot1 = 'previsto'
    xlabel_subplot2 = 'erro'
    ylabel_subplot2 = ylabel_subplot0
    xlabel_subplot3 = ''
    ylabel_subplot3 = xlabel_subplot0
    xlabel = [locals()[f'xlabel_subplot{i}'] for i in range(fig_grade[1])]
    ylabel = [locals()[f'ylabel_subplot{i}'] for i in range(fig_grade[1])]
    list_stats = ['mean', 'range', 'std', 'iqr', 'skew', 'kurtosis']

    common_index = s0.index.intersection(s.index)
    df_s = pd.concat([s0, s], index=1)
    max_value = max(s0.max(), s.max())
    min_value = min(s0.min(), s.min())
    max_abs_error = (s - s0).abs().max
    bins_minamax = range(min_value, max_value * 1.1, (max_value - min_value) / 30)
    bins_absmaxamax = range(-max_abs_error, max_abs_error * 1.1, max_abs_error / 15)
    slope, intercept = np.polyfit(s0, s, 1)

    for k in ['0', '']:
        locals()[f's{k}_'] = locals()[f's{k}']
        locals()[f's{k}'] = locals()[f's{k}_'].loc[common_index]
        locals()[f'{list_stats[0]}_s{k}'] = locals()[f's{k}'].mean()
        locals()[f'{list_stats[1]}_s{k}'] = locals()[f's{k}'].max() - locals()[f's{k}'].min()
        locals()[f'{list_stats[2]}_s{k}'] = locals()[f's{k}'].std()
        locals()[f'{list_stats[3]}_s{k}'] = locals()[f's{k}'].quantile(0.75) - locals()[f's{k}'].quantile(0.25)
        locals()[f'{list_stats[4]}_s{k}'] = locals()[f's{k}'].skew()
        locals()[f'{list_stats[5]}_s{k}'] = locals()[f's{k}'].kurtosis()
    
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
    df_stats = pd.concat([stats_s0, stats_s], index=1)

    corr = df_s.corr().iloc[0,1]
    r2 = r2_score(s, LinearRegression().fit(df_s[s0.name], s).predict(df_s[s0.name]))
    nrmse =   np.sqrt(mean_squared_error(s0, s)) / locals()['range_s0'] if locals()['range_s0'] != 0 else np.nan
    nbias =   (s - s0).mean() / locals()['range_s0'] if locals()['range_s0'] != 0 else np.nan

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], width_ratios=width_ratios)
    axes = []

    for i in range(fig_grade[1]):
        axes.append(fig.add_subplot(gs[i]))

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

    axes[0].axvspan(s0.quantile(0.25), s0.quantile(0.75), color=color_iqr[0], label=f'IQR {xlabel_subplot1}', alpha=0.3)
    axes[0].axvspan(s.quantile(0.25), s.quantile(0.75), color=color_iqr[1], label=f'IQR {ylabel_subplot1}', alpha=0.3)
    axes[0].axvline(x=s0.median(), color=color_linestat['median'][0], linestyle=':', label=f'mediana {xlabel_subplot1} [{s0.median():.2f}]')
    axes[0].axvline(x=s.median(), color=color_linestat['median'][1], linestyle=':', label=f'mediana {ylabel_subplot1} [{s.median():.2f}]')
    axes[0].hist(s0, density=True, bins=bins_minamax, color=color_data[0], label=xlabel_subplot1, alpha=0.5)
    axes[0].hist(s, density=True, bins=bins_minamax, color=color_data[1], label=ylabel_subplot1, alpha=0.5)
    axes[0].axvline(x=stats_s0['mean'], color=color_linestat['mean'][0], linestyle='--', label=f'média {stats_s0.name} [{stats_s0['mean']:.2f}]')
    axes[0].axvline(x=stats_s['mean'], color=color_linestat['mean'][1], linestyle='--', label=f'média {stats_s.name} [{stats_s['mean']:.2f}]')

    axes[1].scatter(s0.values, s.values, color=color_comparison, marker='o', s=3, label='dados cruzados')
    axes[1].plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=1, label=f'identidade')
    axes[1].plot(s0, slope * s0 + intercept, color=color_linestat['R2'], linestyle='--', linewidth=2, label=f'ajuste R² [{r2:.2f}]')

    axes[2].axvspan((s - s0).quantile(0.25), (s - s0).quantile(0.75), color='gray', label=f'IQR erro', alpha=0.3)
    axes[2].axvline(x=(s - s0).median(), color='black', linestyle=':', label=f'mediana erro [{(s - s0).median():.2f}]')
    axes[2].hist(s - s0, bins=bins_absmaxamax, color=color_comparison, label='erro dos dados')
    axes[2].axvline(x=nbias, color=color_linestat['nbias'], linestyle='--', label=f'BIAS normalizado [{nbias:.2f}]')

    box = axes[3].boxplot(df_s, patch_artist=True, labels=df_s.columns)
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(color_data[i])
        patch.set_edgecolor(color_data[i])
        patch.set_linewidth(1.5)  
        for j in range(i * 2, i * 2 + 2):
            box['whiskers'][j].set_color(color_data[i])
            box['whiskers'][j].set_linewidth(1.5) 
        for j in range(i * 2, i * 2 + 2):
            box['caps'][j].set_color(color_data[i])
            box['caps'][j].set_linewidth(1.5) 
        box['medians'][i].set_color(color_linestat['mean'][i])
        box['medians'][i].set_linewidth(3) 
        for flier in box['fliers']:
            flier.set(marker='.', color=color_linestat['mean'][i])

    for ax in axes:
        ax.grid(axis='both')
        ax.legend()

    fig.tight_layout()  
    fig.suptitle(suptitle)  

    return fig
