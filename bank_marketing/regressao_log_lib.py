import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analise_var_numerica_por_percentil(data, x, y, q=10, grafico='none'):
    """
    Ordena a variável x, divide em percentis e sumariza estatísticas.

    Parâmetros:
        data (pd.DataFrame): O banco de dados contendo as variáveis.
        x (str): O nome da variável independente (explanatória).
        y (str): O nome da variável dependente (resposta).
        q (int): O número de percentis (default: 10).
        grafico (str): Opção de gráfico: 'p', 'logito', 'ambos', 'none' (default: 'none').

    Retorno:
        pd.DataFrame: DataFrame com as estatísticas por percentil, incluindo:
                      - Percentil
                      - n (número de linhas)
                      - Min de x
                      - Max de x
                      - p (média de y)
                      - logito de p

    Exemplo de uso
        >> data = pd.DataFrame({'x': np.random.uniform(0, 100, 1000), 
        'y': np.random.randint(0, 2, 1000)})
        >> resultado = analise_var_numerica_por_percentil(data, 'x', 'y', q=10, grafico='ambos')
        >> print(resultado)
    """
    # Certificar-se de que a variável y está no formato numérico
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Ordenar os dados pela variável x
    data = data.sort_values(by=x).reset_index(drop=True)

    # Criar os percentis
    _ , bins_edge = pd.qcut(data[x], q=q, retbins=True, duplicates='drop')
    data['percentil'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, len(bins_edge))], retbins=False, duplicates='drop')

    # Sumarizar as estatísticas por percentil
    summary = data.groupby('percentil').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calcular o logito de p
    summary['logito_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Ajuste para lidar com casos onde p é 0 ou 1
    epsilon = 1e-10  # Pequeno valor para ajustar 0 e 1
    summary['logito_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))


    # Opções de gráfico
    if grafico in ['p', 'logito', 'ambos']:
        plt.figure(figsize=(12, 6))

        if grafico == 'p':
            plt.scatter(summary['percentil'], summary['p'], color='blue')
            plt.title('Gráfico de Percentil x p')
            plt.xlabel('Percentil')
            plt.ylabel('p (média de y)')
            plt.grid(True)
            plt.show()

        elif grafico == 'logito':
            plt.scatter(summary['percentil'], summary['logito_p'], cmap='mako_r')
            plt.title('Gráfico de Percentil x Logito de p')
            plt.xlabel('Percentil')
            plt.ylabel('Logito de p')
            plt.grid(True)
            plt.show()

        elif grafico == 'ambos':
            # Gráficos lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # Gráfico Percentil x p
            axes[0].scatter(summary['percentil'], summary['p'], color='blue')
            axes[0].set_title('Percentil x p')
            axes[0].set_xlabel('Percentil')
            axes[0].set_ylabel('p (média de y)')
            axes[0].grid(True)

            # Gráfico Percentil x Logito de p
            axes[1].scatter(summary['percentil'], summary['logito_p'], cmap='mako_r')
            axes[1].set_title('Percentil x Logito de p')
            axes[1].set_xlabel('Percentil')
            axes[1].set_ylabel('Logito de p')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    return summary


def plot_IC95(model)-> None:

    summary_df = model.summary2().tables[1].reset_index()
    summary_df.rename(columns={"Coef.": "coef", "[0.025": "lower", "0.975]": "upper"}, inplace=True)
    summary_df = summary_df[summary_df["index"] != "Intercept"]  # remove intercepto
    fig, ax = plt.subplots(figsize=(8,4))

    y_positions = range(len(summary_df))

    ax.errorbar(
        summary_df["coef"],
        y_positions,
        xerr=[
            summary_df["coef"] - summary_df["lower"],
            summary_df["upper"] - summary_df["coef"]
        ],
        fmt="o",
        capsize=5,
        ecolor='blue',
        label="Coeficiente"
    )

    ax.axvline(0, color="red", linestyle="--")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df["index"])
    ax.set_title("Intervalos de Confiança dos Coeficientes")

    plt.tight_layout()
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)
    plt.show()

    return None