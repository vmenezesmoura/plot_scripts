import pandas as pd

def fig_2xN_N1subplot(ss: list[pd.Series], xlim_subplot_line0: tuple[pd.Timestamp], ylabel: str, suptitle: str, title: list[str] = None, ylim: tuple[float] = None, xlim_subplot_line1: list[tuple[pd.Timestamp]] = None, subplot_line1_iszoom: bool = False, figsize: tuple[int] = (10, 5), height_ratios: list[float] = [1, 1.5]):

    '''
    Esta função cria uma figura com uma grade de subplots em duas linhas. A primeira linha
    ocupa uma única subplot e exibe todas as séries fornecidas. A segunda linha tem a mesma
    quantidade de subplots que o número de séries, permitindo uma visualização detalhada ou 
    com zoom de cada série individualmente.

    Parâmetros:
    ss (list[pd.Series]): Lista de séries de dados a serem plotadas.
    xlim_subplot_line0 (tuple[pd.Timestamp]): Limite do eixo x para a primeira linha de subplots.
    ylabel (str): Rótulo para o eixo y.
    suptitle (str): Título principal da figura.
    title (list[str], opcional): Lista de títulos para cada subplot. Padrão é None.
    ylim (tuple[float], opcional): Limite do eixo y. Padrão é None.
    xlim_subplot_line1 (list[tuple[pd.Timestamp]], opcional): Limite do eixo x para cada subplot da segunda linha. 
    subplot_line1_iszoom (bool, opcional): Se True, destaca a área de zoom na primeira linha. Padrão é False.
    figsize (tuple[int], opcional): Tamanho da figura. Padrão é (10, 5).
    height_ratios (list[float], opcional): Proporção de altura entre as linhas. Padrão é [1, 1.5].

    Retorna:
    fig (matplotlib.figure.Figure): Objeto da figura criada.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Define a grade da figura como 2 linhas e um número de colunas igual ao número de séries.
    fig_grade = (2, len(ss))
    
    # Lista de cores para destacar regiões de zoom, caso necessário.
    color_emphasis = ['gold', 'deepskyblue', 'orchid', 'yellowgreen', 'grey', 'orangered', 'mediumslateblue']

    # Define os títulos de cada subplot. Se não fornecido, preenche com strings vazias.
    if title is None:
        title = [''] * (fig_grade[1] + 1)
    elif len(title) == 1:
        title = title + [''] * fig_grade[1]
    
    # Se os limites de x para a segunda linha de subplots não forem fornecidos, usa os da primeira linha.
    if xlim_subplot_line1 is not None:
        while len(xlim_subplot_line1) < len(ss):
            xlim_subplot_line1.append(xlim_subplot_line0)

    # Cria a figura e especifica a grade de subplots com a proporção de altura definida.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], height_ratios=height_ratios)
    axes = []

    # Criação dos subplots. A primeira linha tem uma única subplot que se estende por todas as colunas.
    for i in range(fig_grade[0]):
        if i == 0:
            axes.append(fig.add_subplot(gs[i, :]))
            continue
        # A segunda linha contém subplots individuais para cada série.
        for j in range(fig_grade[1]): 
            axes.append(fig.add_subplot(gs[i, j]))

    # Plotagem dos dados em cada subplot.
    for i in range(len(axes)):
        for s in ss:
            axes[i].plot(s, label=s.name)

    # Configuração dos limites e propriedades dos eixos para cada subplot.
    for i in range(len(axes)):
        if i == 0:
            # Limites do eixo x para a primeira linha.
            axes[i].set_xlim(xlim_subplot_line0)
        else:
            if xlim_subplot_line1 is not None:
                # Limites do eixo x para a segunda linha.
                axes[i].set_xlim(xlim_subplot_line1[i - 1])
                if subplot_line1_iszoom:
                    # Se o zoom estiver ativado, destaca a área na primeira linha e ajusta a cor dos eixos.
                    axes[0].axvspan(xlim_subplot_line1[i - 1][0], xlim_subplot_line1[i - 1][1], color=color_emphasis[i - 1], alpha=0.3)
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor(color_emphasis[i - 1]) 
            else:
                # Caso contrário, usa os limites do eixo x da primeira linha.
                axes[i].set_xlim(xlim_subplot_line0)
        
        # Define o rótulo do eixo y e os limites, se fornecidos.
        axes[i].set_ylabel(ylabel)
        if ylim is not None:
            axes[i].set_ylim(ylim)

        # Define o título de cada subplot, adiciona a grade e a legenda.
        axes[i].set_title(title[i])
        axes[i].grid(axis='both')
        axes[i].set_xlabel('')
        axes[i].legend()

    # Ajusta o layout da figura e adiciona o título principal.
    fig.tight_layout()
    fig.suptitle(suptitle)
    
    return fig
