import pandas as pd

def fig_2x2_4subplot(ss0: list[pd.Series], ss1: list[pd.Series], xlim_subplot_line0: tuple[pd.Timestamp], ylabel_subplot_column0: str, suptitle: str, title: list[str] = None, ylim_subplot_column0: tuple[float] = None, ylim_subplot_column1: tuple[float] = None, ylabel_subplot_column1: str = None, xlim_subplot_line1: tuple[pd.Timestamp] = None, subplot_line1_iszoom: bool = False, figsize: tuple[int] = (10, 5), height_ratios: list[float] = [1.5, 1]):
    '''
    Cria uma figura com 4 subplots dispostos em uma grade 2x2, com a possibilidade de zoom
    e diferentes ajustes de eixos e rótulos.

    Parâmetros:
    -----------
    ss0 : list[pd.Series]
        Lista de séries pandas a serem plotadas nos subplots da primeira coluna.
    ss1 : list[pd.Series]
        Lista de séries pandas a serem plotadas nos subplots da segunda coluna.
    xlim_subplot_line0 : tuple[pd.Timestamp]
        Limite do eixo x para os subplots da primeira linha.
    ylabel_subplot_column0 : str
        Rótulo do eixo y para os subplots da primeira coluna.
    suptitle : str
        Título geral da figura.
    title : list[str], opcional
        Lista de títulos para cada subplot. Se não for fornecido, uma lista de strings vazias será usada.
    ylim_subplot_column0 : tuple[float], opcional
        Limite do eixo y para os subplots da primeira coluna.
    ylim_subplot_column1 : tuple[float], opcional
        Limite do eixo y para os subplots da segunda coluna.
    ylabel_subplot_column1 : str, opcional
        Rótulo do eixo y para os subplots da segunda coluna. Se não fornecido, usará o mesmo rótulo da primeira coluna.
    xlim_subplot_line1 : tuple[pd.Timestamp], opcional
        Limite do eixo x para os subplots da segunda linha. Se não for fornecido, usará os limites da primeira linha.
    subplot_line1_iszoom : bool, opcional
        Se True, aplica zoom nos subplots da segunda linha e destaca as áreas correspondentes na linha de cima.
    figsize : tuple[int], opcional
        Tamanho da figura em polegadas. Padrão é (10, 5).
    height_ratios : list[float], opcional
        Proporção da altura das linhas dos subplots. Padrão é [1.5, 1].

    Retorna:
    --------
    fig : matplotlib.figure.Figure
        Objeto figura contendo os 4 subplots configurados.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Definição das configurações estáticas da função
    fig_grade = (2, 2)  # Define o layout da grade: 2 linhas por 2 colunas
    color_emphasis = 'gold'  # Cor para realçar as áreas de zoom

    # Verifica os parâmetros de entrada e ajusta valores default quando necessário
    if title is None:
        title = [''] * (fig_grade[0] * fig_grade[1])  # Se nenhum título for fornecido, cria uma lista de strings vazias
    elif len(title) == fig_grade[1]:
        title = title + [''] * fig_grade[1]  # Se o título for fornecido parcialmente, completa a lista
    if ylabel_subplot_column1 is None:
        ylabel_subplot_column1 = ylabel_subplot_column0  # Usa o mesmo rótulo se o segundo não for fornecido

    # Cria a figura com o tamanho definido e uma grade de subplots usando GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(fig_grade[0], fig_grade[1], height_ratios=height_ratios)  # Define a proporção das alturas das linhas
    axes = []

    # Adiciona os subplots à figura e plota as séries fornecidas em cada um
    for i in range(fig_grade[0]):  # Laço para as linhas
        for j in range(fig_grade[1]):  # Laço para as colunas
            axes.append(fig.add_subplot(gs[i, j]))  # Cria cada subplot
            # Verifica as séries a serem plotadas na primeira coluna
            if len(axes)%2 == 1:
                for s in ss0:
                    axes[-1].plot(s, label=s.name)  # Plota cada série no subplot atual
            # Verifica as séries a serem plotadas na segunda coluna
            if len(axes)%2 == 0:
                for s in ss1:
                    axes[-1].plot(s, label=s.name)  # Plota cada série no subplot atual

    # Ajusta os limites dos eixos e outras configurações específicas para cada subplot
    for i in range(len(axes)):
        # Ajusta o limite x (xlim) dos subplots da primeira linha (linha 0)
        if i // fig_grade[0] == 0:
            axes[i].set_xlim(xlim_subplot_line0)
        # Para os subplots da segunda linha (linha 1), verifica se há limites específicos
        elif i // fig_grade[0] == 1:
            if xlim_subplot_line1 is not None:
                axes[i].set_xlim(xlim_subplot_line1)
                # Se estiver aplicando zoom, adiciona destaque visual nos subplots da linha anterior
                if subplot_line1_iszoom:
                    axes[i - fig_grade[1]].axvspan(xlim_subplot_line1[0], xlim_subplot_line1[1], color=color_emphasis, alpha=0.3)
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor(color_emphasis)  # Ajusta a cor das bordas do subplot com zoom
            else:
                axes[i].set_xlim(xlim_subplot_line0)
        
        # Ajusta o rótulo do eixo y (ylabel) e os limites y (ylim) dependendo da coluna do subplot
        if i % fig_grade[1] == 0:  # Subplots na primeira coluna
            axes[i].set_ylabel(ylabel_subplot_column0)
            if ylim_subplot_column0 is not None:
                axes[i].set_ylim(ylim_subplot_column0)
        elif i % fig_grade[1] == 1:  # Subplots na segunda coluna
            axes[i].set_ylabel(ylabel_subplot_column1)
            if ylim_subplot_column1 is not None:
                axes[i].set_ylim(ylim_subplot_column1)
        
        # Define o título de cada subplot
        axes[i].set_title(title[i])
        axes[i].grid(axis='both')  # Ativa a grade em ambos os eixos
        axes[i].set_xlabel('')  # Limpa o rótulo do eixo x (pode ser configurado depois)
        axes[i].legend()  # Exibe a legenda para as séries

    # Ajusta o layout da figura para evitar sobreposição entre os subplots
    fig.tight_layout()
    
    # Define o título geral da figura
    fig.suptitle(suptitle)
    
    return fig  # Retorna a figura criada
