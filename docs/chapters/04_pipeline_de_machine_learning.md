## 4. Pipeline de Machine Learning

### 4.1. Formulação do target

O target principal da solução é:

- `target_future_risk`

Regra:

- `1` se `defasagem_{t+1} < 0`
- `0` caso contrário

Assim, o modelo tenta antecipar risco futuro, e não reconstruir apenas o estado atual.

### 4.2. Estratégias de features

O projeto compara duas estratégias de entrada para os modelos estatísticos.

### 4.2.1. O que foi feito de feature engineering

No projeto, o trabalho de feature engineering ficou concentrado em cinco decisões principais:

1. padronização das colunas para um schema canônico único, permitindo que os mesmos modelos consumam arquivos de anos diferentes
2. separação explícita entre variáveis numéricas e categóricas, para que cada tipo receba o tratamento correto no pipeline
3. definição de dois conjuntos de entrada, `strategy_a` e `strategy_b`, para comparar uma abordagem mais enxuta com outra mais completa
4. criação da variável derivada `flag_ipp_ausente`, que informa quando o indicador `IPP` está ausente
5. montagem dos pares temporais aluno-ano, de forma que as variáveis do ano atual sejam usadas para prever o risco do ano seguinte

Em termos práticos, isso significa que o projeto não tentou criar dezenas de variáveis artificiais. A ênfase foi em organizar melhor as variáveis existentes, selecionar conjuntos coerentes para comparação e registrar de forma explícita quando a ausência de um indicador também pode carregar informação útil.

#### Estratégia A

Variáveis:

- `fase`
- `turma`
- `idade`
- `genero`
- `ano_ingresso`
- `instituicao_ensino`
- `numero_avaliacoes`
- `inde_atual`
- `iaa`
- `ieg`
- `ips`
- `ida`
- `nota_matematica`
- `nota_portugues`
- `nota_ingles`
- `ipv`

Uso:

- cenário mais conservador, com sinais educacionais e de acompanhamento sem ampliar tanto o conjunto de contexto

#### Estratégia B

Inclui tudo da Estratégia A e adiciona:

- `ian`
- `fase_ideal`
- `ipp`
- `flag_ipp_ausente`

Uso:

- cenário mais completo, com fotografia educacional mais rica do estudante

### 4.3. Modelos comparados

Os candidatos comparados foram:

- `baseline_persistencia`
- `logistic_regression`
- `random_forest`
- `hist_gradient_boosting`

O baseline usa a persistência da `defasagem` atual como régua de comparação.

Em termos simples, o baseline é a solução mais simples aceitável para o problema. Ele existe para responder a uma pergunta importante:

> Os modelos mais sofisticados realmente acrescentam valor ou estão apenas tentando complicar algo que uma regra simples já faz?

Neste projeto, o `baseline_persistencia` funciona assim:

- se o estudante já apresenta defasagem no ano atual, o baseline prevê risco no ano seguinte
- se o estudante não apresenta defasagem no ano atual, o baseline prevê ausência de risco no ano seguinte

Portanto, ele não é um modelo “inteligente” no sentido de aprender padrões complexos. Ele é uma regra simples, usada como referência mínima de comparação.

### 4.4. Validação e seleção

Foi adotada validação temporal:

- treino com pares `2022 -> 2023`
- validação com pares `2023 -> 2024`

Métricas principais:

- `F2-score`
- `Recall`
- `Precision`
- `PR-AUC`
- `ROC-AUC`
- `Brier Score`

A métrica principal de seleção foi o `F2-score`, porque ela dá mais peso ao `Recall`. Isso faz sentido neste problema, em que deixar de identificar um estudante vulnerável tende a ser mais grave do que sinalizar alguns casos adicionais.

### 4.5. Pós-processamento do treino

Após o treino:

- o melhor bundle estatístico é serializado em `joblib`
- os bundles candidatos também são salvos
- os modelos segmentados de transição são salvos como camada complementar
- o histórico do treino é persistido em banco
- a base de referência para drift é persistida
- um relatório exportável é gerado a partir do registro do treino

### 4.6. Camada complementar de scores analógicos

A solução mantém a saída binária principal do case, mas adiciona uma camada analógica complementar.

Na prática, cada predição agora pode devolver:

- `prediction_risco_defasagem_futura`: decisão binária principal
- `probabilidade_risco_defasagem_futura`: probabilidade geral de risco futuro
- `grupo_baseline_atual`: indica se o estudante já está ou não defasado hoje
- `score_entrada_em_risco`: para estudantes que hoje não estão defasados
- `score_recuperacao`: para estudantes que hoje já estão defasados
- `score_persistencia_defasagem`: complemento do score de recuperação

Essa camada foi criada para responder melhor a perguntas operacionais do tipo:

- quem tem mais chance de piorar?
- quem tem mais chance de se recuperar?

Ou seja, a solução deixou de ser apenas binária na leitura operacional, mas sem abandonar a resposta binária principal exigida pelo problema.

---
