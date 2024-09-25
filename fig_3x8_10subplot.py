import pandas as pd

def fig_3x4_10subplot(ss0: list[pd.Series], ss1: list[pd.Series], xlim_subplot_line0: tuple[pd.Timestamp], ylabel_subplot_column0: str, suptitle: str, title: list[str] = None, ylim_subplot_column0: tuple[float] = None, ylim_subplot_column1: tuple[float] = None, ylabel_subplot_column1: str = None, xlim_subplot_line12: list[tuple[pd.Timestamp]] = None, figsize: tuple[int] = (10, 7), height_ratios: list[float] = [1, 1.5, 1.5]):
    '''
    Esta função cria uma figura com uma grade de subplots 3x4, totalizando 10 subplots para visualização de séries temporais.

    Parâmetros:
    - ss0: lista de séries temporais a serem plotadas nos subplots da coluna 0.
    - ss1: lista de séries temporais a serem plotadas nos subplots da coluna 1.
    - xlim_subplot_line0: limite x para o eixo x nos subplots da linha 0.
    - ylabel_subplot_column0: rótulo do eixo y para os subplots da coluna 0.
    - suptitle: título geral da figura.
    - title: lista de títulos para cada subplot. Se None, todos os subplots terão título vazio.
    - ylim_subplot_column0: limite y para os subplots da coluna 0. Se None, o limite será automático.
    - ylim_subplot_column1: limite y para os subplots da coluna 1. Se None, o limite será automático.
    - ylabel_subplot_column1: rótulo do eixo y para os subplots da coluna 1. Se None, será igual a `ylabel_subplot_column0`.
    - xlim_subplot_line12: lista de limites x para cada subplot das linhas 1 e 2. Se None, será igual a `xlim_subplot_line0`.
    - figsize: tamanho da figura (largura, altura).
    - height_ratios: proporção da altura para cada linha na grade de subplots.

    Retorna:
    - fig: objeto de figura do Matplotlib contendo os subplots gerados.

    Descrição:
    A função organiza a figura em uma grade de subplots de 3 linhas e 4 colunas:
    - A linha 0 (superior) possui 2 subplots abrangendo 2 colunas cada.
    - As linhas 1 e 2 contêm 4 subplots cada, dispostos de forma uniforme.

    - As séries em `ss0` são plotadas nos subplots com índices em `i0`, enquanto `ss1` nos índices em `i1`.
    - O parâmetro `xlim_subplot_line12` permite ajustar os limites do eixo x para subplots específicos, adicionando destaques (axvspan) nos subplots da linha 0, conforme necessário.
    - A função ajusta o layout dos subplots para evitar sobreposição de elementos.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig_grade = (3, 4)  # Dimensão da grade de subplots (3 linhas, 4 colunas)
    color_emphasis = ['gold', 'deepskyblue', 'orchid', 'yellowgreen']  # Cores para destaques nos subplots
    i0 = [0, 2, 3, 6, 7]  # Índices dos subplots para a primeira lista de séries (`ss0`)
    i1 = [1] + [x + 2 for x in i0[1:]]  # Índices dos subplots para a segunda lista de séries (`ss1`)

    # Configuração dos títulos dos subplots
    if title is None:
        title = [''] * (fig_grade[0] * fig_grade[1] - 2)  # Títulos padrão vazios
    elif len(title) == fig_grade[1] // 2:
        title = title + [''] * (fig_grade[1] * (fig_grade[0] - 1))  # Adiciona títulos vazios caso não tenha suficiente
    if ylabel_subplot_column1 is None:
        ylabel_subplot_column1 = ylabel_subplot_column0  # Rótulo padrão do eixo y para a coluna 1
    if len(xlim_subplot_line12) == fig_grade[1] // 2:
        span_subplot_line2 = False
        xlim_subplot_line12 = xlim_subplot_line12 + [xlim_subplot_line0] * 2  # Ajusta limites x para linha 2
    else:
        span_subplot_line2 = True

    # Criação da figura e grade de subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], height_ratios=height_ratios)
    axes = []

    # Adiciona os subplots à figura
    for i in range(fig_grade[0]):
        if i == 0:
            # Primeira linha (2 subplots abrangendo 2 colunas cada)
            axes.append(fig.add_subplot(gs[i, :fig_grade[1]//2]))
            axes.append(fig.add_subplot(gs[i, fig_grade[1]//2:]))
            continue
        for j in range(fig_grade[1]):
            # Linhas 1 e 2 (subplots simples)
            axes.append(fig.add_subplot(gs[i, j]))

    # Plota as séries temporais nos subplots correspondentes
    for i in range(len(axes)):
        if i in i0:
            for s in ss0:
                axes[i].plot(s, label=s.name)
        elif i in i1:
            for s in ss1:
                axes[i].plot(s, label=s.name)

    # Configura os limites dos eixos x e y, títulos e outras propriedades dos subplots
    for i in range(len(axes)):
        if i // 2 == 0:
            axes[i].set_xlim(xlim_subplot_line0)
        else:
            if xlim_subplot_line12 is not None:
                if i in i0[1:]:
                    j = i0[1:].index(i)
                    axes[i].set_xlim(xlim_subplot_line12[j])
                    if span_subplot_line2 and j >= 2:
                        axes[0].axvspan(xlim_subplot_line12[j][0], xlim_subplot_line12[j][1], color=color_emphasis[j], alpha=0.3)
                        for spine in axes[i].spines.values():
                            spine.set_edgecolor(color_emphasis[j]) 
                elif i in i1[1:]:
                    j = i1[1:].index(i)
                    axes[i].set_xlim(xlim_subplot_line12[j])
                    if span_subplot_line2 and j >= 2:
                        axes[1].axvspan(xlim_subplot_line12[j][0], xlim_subplot_line12[j][1], color=color_emphasis[j], alpha=0.3)
                        for spine in axes[i].spines.values():
                            spine.set_edgecolor(color_emphasis[j]) 
            else:
                axes[i].set_xlim(xlim_subplot_line0)
        
        if i in i0:
            axes[i].set_ylabel(ylabel_subplot_column0)
            if ylim_subplot_column0 is not None:
                axes[i].set_ylim(ylim_subplot_column0)
        elif i in i1:
            axes[i].set_ylabel(ylabel_subplot_column1)
            if ylim_subplot_column1 is not None:
                axes[i].set_ylim(ylim_subplot_column1)
        
        axes[i].set_title(title[i])
        axes[i].grid(axis='both')
        axes[i].set_xlabel('')  # Oculta rótulos do eixo x por padrão
        axes[i].legend()

    fig.tight_layout()  # Ajusta o layout da figura para evitar sobreposição
    fig.suptitle(suptitle)  # Define o título principal da figura
    
    return fig  # Retorna a figura gerada
