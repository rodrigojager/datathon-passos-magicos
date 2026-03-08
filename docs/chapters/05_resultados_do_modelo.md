## 5. Resultados do Modelo

### 5.1. Resultado operacional atual

No treino mais recente, o modelo estatístico definido como padrão da API foi:

- modelo: `random_forest`
- estratégia: `strategy_b`
- versão dos dados: `v3-r3030`

Os demais candidatos treinados continuam disponíveis para uso manual na tela de `Predição`, por meio do dropdown de seleção de modelo. Em outras palavras:

- existe um modelo padrão, usado quando nenhum candidato é escolhido explicitamente
- e existem modelos adicionais disponíveis para comparação prática e uso manual

Métricas:

- `F2`: `0.5595`
- `Recall`: `0.5649`
- `Precision`: `0.5387`
- `PR-AUC`: `0.5933`
- `ROC-AUC`: `0.6939`
- `Brier`: `0.2173`

### 5.2. Comparação resumida entre candidatos

| Modelo | Estratégia | F2 | Recall | Precision | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| baseline_persistencia | baseline_current_defasagem | 0.6884 | 0.7273 | 0.5671 | 0.5222 | 0.6765 |
| logistic_regression | strategy_a | 0.1401 | 0.1234 | 0.3065 | 0.3811 | 0.4925 |
| random_forest | strategy_a | 0.4917 | 0.4805 | 0.5421 | 0.4623 | 0.6372 |
| hist_gradient_boosting | strategy_a | 0.4719 | 0.4643 | 0.5053 | 0.4500 | 0.6113 |
| logistic_regression | strategy_b | 0.1391 | 0.1201 | 0.3776 | 0.3973 | 0.5139 |
| random_forest | strategy_b | 0.5595 | 0.5649 | 0.5387 | 0.5933 | 0.6939 |
| hist_gradient_boosting | strategy_b | 0.4652 | 0.4513 | 0.5305 | 0.4923 | 0.6470 |

### 5.3. Interpretação dos resultados

O baseline aparece muito forte. Isso é esperado porque a defasagem tem persistência temporal relevante.

Mesmo assim, a solução faz uma distinção metodológica importante:

- o baseline é mantido como referência de negócio e auditoria
- o melhor modelo estatístico vira o modelo padrão da API
- os demais candidatos treinados permanecem acessíveis para predição manual

O resultado também confirmou a hipótese técnica mais provável para esse tipo de dado tabular:

- modelos baseados em árvore superaram a regressão logística
- o `random_forest` foi o melhor modelo estatístico na comparação atual

### 5.4. Estudo residual do baseline

Além da comparação entre modelos, a solução executa um estudo específico sobre os casos em que o baseline falha.

Na validação temporal mais recente, o baseline apresentou:

- `510` casos corretos
- `84` falsos negativos
- `171` falsos positivos

Os dois grupos mais importantes nesse estudo são:

- `falsos negativos`: estudantes que não estavam defasados no ano atual, mas entraram em risco no ano seguinte
- `falsos positivos`: estudantes que estavam defasados no ano atual, mas não permaneceram em risco no ano seguinte

Nos falsos negativos, as variáveis que mais apareceram como relevantes no estudo complementar foram:

- `fase`
- `fase_ideal`
- `ano_ingresso`
- `ipv`
- `instituicao_ensino`
- `ipp`
- `turma`
- `idade`
- `ida`

Isso sugere que a entrada em risco, mesmo sem uma defasagem atual explícita, parece estar associada a contexto escolar, estágio da trajetória e sinais de acompanhamento.

Nos falsos positivos, que representam casos de recuperação apesar da expectativa pessimista do baseline, apareceram com mais peso:

- `fase`
- `ipp`
- `fase_ideal`
- `inde_atual`
- `ipv`
- `ida`
- `turma`
- `nota_portugues`
- `nota_matematica`
- `numero_avaliacoes`

Isso sugere que a recuperação pode estar ligada a melhora de desempenho, maior acompanhamento e melhor aderência à fase esperada.

Esse estudo é importante porque ajuda a responder uma pergunta prática:

> Em que situações o baseline deixa de ser suficiente?

Ele também abre uma linha de evolução para a solução (vide item 5.5):

- um modelo específico para estudantes que ainda não estão defasados, mas podem entrar em risco
- um modelo específico para estudantes já defasados que podem se recuperar

### 5.5. Resultados da camada de scores de transição

Com base nesse aprendizado, a solução passou a treinar dois modelos segmentados:

- `entrada em risco`: aplicado a estudantes que hoje não estão defasados
- `recuperação`: aplicado a estudantes que hoje já estão defasados

No estado atual, os resultados dessa camada ficaram assim:

#### Modelo de entrada em risco

- `F2`: `0.0605`
- `Recall`: `0.0595`
- `Precision`: `0.0649`
- `PR-AUC`: `0.2415`
- `ROC-AUC`: `0.5912`

Leitura:

- esse modelo ainda está fraco e não deve ser tratado como substituto do modelo principal
- ele funciona mais como score exploratório do que como decisão confiável isolada

#### Modelo de recuperação

- `F2`: `0.4883`
- `Recall`: `0.4620`
- `Precision`: `0.6320`
- `PR-AUC`: `0.6399`
- `ROC-AUC`: `0.7277`

Leitura:

- esse modelo ficou bem mais promissor
- a recuperação parece ser um fenômeno mais capturável pelos dados atuais do que a entrada inesperada em risco

Conclusão prática:

- a camada analógica segmentada é útil para priorização e leitura de trajetória
- ela ainda não substitui a comparação global com o baseline
- o maior potencial atual está em identificar recuperação e persistência, mais do que entrada em risco

---
