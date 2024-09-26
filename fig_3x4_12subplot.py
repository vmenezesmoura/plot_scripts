import pandas as pd

def fig_3x4_4subplot(ss0: list[pd.Series], ss: list[pd.Series], xlabel_subplot0:list[str], title:list[str], suptitle: str, figsize: tuple[int] = (8, 10), width_ratios: list[float] = [2, 1.5, 1, 1]):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    fig_grade = (3, 4)
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

    if len(ss0) != len(ss):
        raise ValueError('ss0 tem quantidade de séries para comparação diferente de ss')
    while len(xlabel_subplot0) < len(ss0):
        if len(xlabel_subplot0) == 1:
            xlabel_subplot0 = xlabel_subplot0 * 3
        else:
            xlabel_subplot0.append('')
    while len(title) < len(ss0):
        if len(title) == 1:
            title = title * 3
        else:
            title.append('')

    df_ss = []
    max_values = []
    min_values = []
    max_abs_errors = []
    bins_minamaxs = []
    bins_absmaxamaxs = []
    slopes, intercepts = [], []
    stats_s0s = []
    stats_ss = []
    df_statss = []
    corrs = []
    r2s = []
    nrmses = []
    nbiass = []
    for i in range(len(ss0)):
        s0 = ss0[i]
        s = ss[i]
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

        df_ss.append(df_s)
        max_values.append(max_value)
        min_values.append(min_value)
        max_abs_errors.append(max_abs_error)
        bins_minamaxs.append(bins_minamax)
        bins_absmaxamaxs.append(bins_absmaxamax)
        slopes.append(slope)
        intercepts.append(intercept)
        stats_s0s.append(stats_s0)
        stats_ss.append(stats_s)
        df_statss.append(df_stats)
        corrs.append(corr)
        r2s.append(r2)
        nrmses.append(nrmse)
        nbiass.append(nbias)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], width_ratios=width_ratios)
    axes = []

    for i in range(fig_grade[0]):
        axes.append([])
        for j in range(fig_grade[1]):
            axes[i].append(fig.add_subplot(gs[i, j]))

    for i in range(fig_grade[0]):
        s0 = ss0[i]
        s = ss[i]

        axes[i][0].set_title(f'Distribuição {xlabel_subplot1} vs {ylabel_subplot1} [{title[i]}]')
        axes[i][0].set_xlabel(xlabel_subplot0[i])
        axes[i][0].set_ylabel(ylabel_subplot0)

        axes[i][1].set_title(f'Correlação')
        axes[i][1].set_xlabel(xlabel_subplot1)
        axes[i][1].set_ylabel(ylabel_subplot1)

        axes[i][2].set_title(f'Erro')
        axes[i][2].set_xlabel(xlabel_subplot2)
        axes[i][2].set_ylabel(ylabel_subplot2)

        axes[i][3].set_title(f'Boxplot')
        axes[i][3].set_xlabel(xlabel_subplot3)
        axes[i][3].set_ylabel(ylabel_subplot3)

        axes[i][0].axvspan(s0.quantile(0.25), s0.quantile(0.75), color=color_iqr[0], label=f'IQR {xlabel_subplot1}', alpha=0.3)
        axes[i][0].axvspan(s.quantile(0.25), s.quantile(0.75), color=color_iqr[1], label=f'IQR {ylabel_subplot1}', alpha=0.3)
        axes[i][0].axvline(x=s0.median(), color=color_linestat['median'][0], linestyle=':', label=f'mediana {xlabel_subplot1} [{s0.median():.2f}]')
        axes[i][0].axvline(x=s.median(), color=color_linestat['median'][1], linestyle=':', label=f'mediana {ylabel_subplot1} [{s.median():.2f}]')
        axes[i][0].hist(s0, density=True, bins=bins_minamaxs[i], color=color_data[0], label=xlabel_subplot1, alpha=0.5)
        axes[i][0].hist(s, density=True, bins=bins_minamaxs[i], color=color_data[1], label=ylabel_subplot1, alpha=0.5)
        axes[i][0].axvline(x=stats_s0s[i]['mean'], color=color_linestat['mean'][0], linestyle='--', label=f'média {stats_s0s[i].name} [{stats_s0s[i]['mean']:.2f}]')
        axes[i][0].axvline(x=stats_ss[i]['mean'], color=color_linestat['mean'][1], linestyle='--', label=f'média {stats_ss[i].name} [{stats_ss[i]['mean']:.2f}]')

        axes[i][1].scatter(s0.values, s.values, color=color_comparison, marker='o', s=3, label='dados cruzados')
        axes[i][1].plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=1, label=f'identidade')
        axes[i][1].plot(s0, slopes[i] * s0 + intercepts[i], color=color_linestat['R2'], linestyle='--', linewidth=2, label=f'ajuste R² [{r2s[i]:.2f}]')

        axes[i][2].axvspan((s - s0).quantile(0.25), (s - s0).quantile(0.75), color='gray', label=f'IQR erro', alpha=0.3)
        axes[i][2].axvline(x=(s - s0).median(), color='black', linestyle=':', label=f'mediana erro [{(s - s0).median():.2f}]')
        axes[i][2].hist(s - s0, bins=bins_absmaxamaxs[i], color=color_comparison, label='erro dos dados')
        axes[i][2].axvline(x=nbiass[i], color=color_linestat['nbias'], linestyle='--', label=f'BIAS normalizado [{nbiass[i]:.2f}]')

        df_ss[i].plot(kind='box', vert=1, widths=0.5, grid=0, ax=axes[i][3], color=dict(boxes=color_data, whiskers=color_data, medians=color_linestat['median'], caps=color_data))

        for ax in axes[i]:
            ax.grid(axis='both')
            ax.legend()

   
    fig.suptitle(suptitle)
    fig.tight_layout()    

    df_concatenado = pd.concat([df.add_suffix(sufixo) for df, sufixo in zip(df_ss, ['.1', '.2', '.3'])], axis=1)

    return fig, df_concatenado
