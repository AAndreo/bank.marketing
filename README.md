# Bank Marketing | Avaliando a qualidade do modelo de regressão para diferentes estratégias.

<img width="80%" src="img/capa-marketing-bancario.png">

## Contexto
Descubra as melhores estratégias para aprimorar a próxima campanha de marketing. Como a instituição financeira pode aumentar a eficácia de suas futuras campanhas de marketing? Para responder a essa pergunta, precisamos analisar a última campanha de marketing realizada pelo banco e identificar os padrões que nos ajudarão a chegar a conclusões para o desenvolvimento de estratégias futuras.

### Nossos dados
Os dados estão relacionados com campanhas de marketing direto de uma instituição bancária portuguesa. As campanhas de marketing baseavam-se em chamadas telefónicas. Muitas vezes, era necessário mais do que um contato com o mesmo cliente, para apurar se o produto (depósito a prazo bancário) seria ('sim') ou não ('não') assinado.

### O que precisa ser feito?
O objetivo da classificação é prever se o cliente irá assinar (sim/não) um depósito a prazo (variável y) e apresentar um modelo de regressão logística avaliando sua qualidade em diferentes tipos de objetivo : 
- decisão
- ordenação
- estimação

### Qual é a história que os dados nos contaram?
- Com a interpretação das Odds Ratios, chegamos a essas conclusões:

    - Clientes com ensino secundário tem 22% menos chance de assinar um depósito a prazo do que clientes com ensino ensino superior ou desconhecido.
    - Clientes com ensino primário tem 34,86% menos chance de assinar um depósito a prazo do que clientes com ensino ensino superior ou desconhecido.
    - Clientes que foram contatados pelo tipo de comunicação desconhecido tem 72,73% menos chance de assinar um depósito a prazo do que clientes contatados através de celular ou telefone.
    - Clientes com empréstimo habitacional tem 41,60% menos chance de assinar um depósito a prazo do que clientes sem empréstimo habitacional.
    - Clientes aposentados tem 124,88% mais chance de assinar um depósito a prazo do que clientes com cargo de gerenciamento ou outros.
    - Clientes com empréstimo pessoal tem 51,28% menos chance de assinar um depósito a prazo do que clientes sem empréstimo pessoal.
    - Clientes divorciados ou viuvos tem 57,95% mais chance de assinar um depósito a prazo do que clientes casados.
    - Clientes solteiros tem 33,31% mais chance de assinar um depósito a prazo do que clientes casados.
    - Clientes cujo ultimo contato do ano foi no 3o. trimestre tem 29,41% menos chance de assinar um depósito a prazo do que clientes cujo ultimo contato do ano foi no 1o. ou 2o. ou 4o. trimestre.
    - Clientes onde o resultado da campanha de marketing anterior foi um sucesso tem 1129% mais chance de assinar um depósito a prazo do que clientes onde o resultado da campanha foi um fracasso ou desconhecido.
    - Clientes onde o resultado da campanha de marketing anterior foi outro tem 69,25% mais chance de assinar um depósito a prazo do que clientes onde o resultado da campanha foi um fracasso ou desconhecido.
    - Clientes que foram contatados entre 2 e 25 vezes em uma campanha anterior tem 110,28% mais chance de assinar um depósito a prazo do que clientes que não foram contatados ou foram contatados uma unica vez.
    - Clientes no qual o número de dias que se passaram do ultimo contato em uma campanha anterior foi entre 93 e 871 dias, possuem 41,13% menos chance de assinar um depósito a prazo do que clientes que não foram    contatados ou o número de dias transcorridos foi de até 92.
    - Para cada aumento de uma unidade no número de contatos realizados durante essa campanha, a chance do cliente assinar um depósito a prazo diminuem 5,92%, mantendo constantes as demais variáveis.
    - Para cada aumento de uma unidade na duração do ultimo contato, em segundos, a chance do cliente assinar um depósito a prazo aumenta em 0,4%, mantendo constantes as demais variáveis.

    ### Qual é a qualidade do nosso modelo para cenários diferentes?

    #### Primeiro cenário - Objetivo Decisão
- A empresa decidiu que probabilidades maiores que 0.5 (p_chapeu > 0.5) serão considerados como casos que o cliente assinou um depósito a prazo (1), caso contrário não assinou (0).

- O volume de clientes para esse cenário é de 240 clientes assinaram o depósito a prazo (1) e 4281 clientes não assinaram o depósito a prazo (0).

- Baseado no resultado da precisão de corte de 0.5, de todos os clientes que o modelo disse que vão assinar o depósito a prazo, o acerto foi de 66,25%, e de todos os clientes que realmente assinaram o depósito a prazo, o modelo teve exito de 30,52% no alcance.

    #### Segundo cenário - Objetivo estimação
- A empresa gostaria de saber, qual a esperança de valores de depositos a prazo receberia na próxima campanha?

- O volume de clientes para esse cenário é de 521 clientes assinaram o depósito a prazo (1) e 4000 clientes não assinaram o depósito a prazo (0).
- Vamos assumir que o valor médio dos depósitos a prazo seja de 775 euros.
- A esperança de valores de depósitos a prazo atráves do modelo seria de 403775.00 euros.

    #### Terceiro cenário - Objetivo ordenação
- A empresa gostaria de atuar de uma forma mais ativa numa próxima campanha, nos 20% dos clientes com maior probabilidade de assinar um depósito a prazo.

- Restrição de negócio: capacidade de atuar em somente 20% dos clientes.
- Se a empresa não utilizar o modelo, terá que selecionar os clientes de forma aleatória e tera um retorno de 11,52%, ou seja, de todos os clientes abordados, somente 11,52% deles assinariam um depósito a prazo.
- Utilizando o modelo, dentro da restrição de 20% da operação, consegue-se trazer um retorno 261,48% maior do que não ter o modelo.
