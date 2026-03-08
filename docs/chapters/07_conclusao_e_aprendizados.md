## 7. Conclusão e Aprendizados

### 7.1. O que o projeto mostrou

O projeto confirmou que é possível estruturar uma solução de Machine Learning útil para apoiar a Passos Mágicos na priorização de estudantes com maior risco de defasagem futura.

Do ponto de vista técnico, os resultados mostraram três aprendizados importantes:

1. existe persistência temporal relevante da defasagem
2. modelos baseados em árvore se adaptam melhor a esse conjunto de dados do que uma regressão logística simples
3. ampliar a visão do estudante com a `strategy_b` melhora a capacidade de previsão do melhor modelo estatístico
4. a camada segmentada de transição é mais promissora para recuperação do que para entrada inesperada em risco

### 7.2. Variáveis que parecem mais relevantes para o diagnóstico

A análise foi baseada no modelo padrão da API, `random_forest` com `strategy_b`, complementada por comparação descritiva entre estudantes que ficaram em risco e estudantes que não ficaram em risco no ano seguinte.

Quando o foco é apenas previsão, apareceram com bastante peso variáveis como:

- `inde_atual`
- `ipv`
- `ian`
- `ipp`
- `ida`
- `nota_matematica`
- `nota_portugues`
- `ieg`
- `numero_avaliacoes`
- `fase` e `fase_ideal`
- `instituicao_ensino`

Mas essa lista mistura dois tipos de sinal:

- variáveis úteis para previsão
- variáveis úteis para intervenção

Para uso operacional da Passos Mágicos, a leitura mais importante é a das variáveis acionáveis, isto é, aquelas sobre as quais a equipe pode agir mais diretamente.

As variáveis acionáveis mais úteis na leitura final foram:

- `ida`
- `nota_matematica`
- `nota_portugues`
- `nota_ingles`
- `ieg`
- `numero_avaliacoes`
- `iaa`
- `ips`
- `ipp`
- `ipv`

Em termos simples, a piora e a melhora parecem estar mais ligadas a:

- desempenho educacional global
- desempenho acadêmico em disciplinas
- engajamento e acompanhamento
- suporte psicossocial e psicopedagógico

Já variáveis como `fase`, `fase_ideal`, `IAN` e `defasagem` foram mantidas como contexto, mas não devem ser tratadas como base principal de interpretação, porque elas descrevem muito diretamente o próprio atraso escolar.

### 7.3. Variáveis acionáveis mais associadas a maior chance de piora

Na análise descritiva da base consolidada, os estudantes que ficaram em risco no ano seguinte apresentaram, em média:

- `IPV` mais baixo
- `IPP` mais baixo
- `IDA` mais baixo
- notas mais baixas em matemática, português e inglês
- `IEG` mais baixo
- `IAA` mais baixo
- `IPS` mais baixo
- menor número de avaliações registradas

Esses sinais sugerem que a chance de defasagem não está ligada a um único fator isolado. Ela parece surgir de um acúmulo de fragilidades acadêmicas e de acompanhamento.

### 7.4. Variáveis acionáveis mais associadas a maior chance de melhora

Nos casos de recuperação apesar da expectativa pessimista do baseline, apareceram com mais força:

- `IPV` mais alto
- `IPP` mais alto
- `IDA` mais alto
- notas mais altas nas disciplinas
- `IEG` mais alto
- `IAA` mais alto
- `IPS` mais alto
- maior número de avaliações

Em leitura prática, isso sugere que a melhora está associada a uma trajetória mais consistente de:

- desempenho acadêmico
- engajamento
- acompanhamento
- suporte psicossocial e psicopedagógico

### 7.5. Pontos que exigem cautela

Nem toda variável com importância alta deve ser lida como causa direta.

Exemplos:

- `fase`
- `fase_ideal`
- `ian`
- `defasagem`
- `instituicao_ensino`
- `idade`

Essas variáveis podem funcionar como marcadores de contexto ou proxies do próprio atraso escolar. Elas são úteis para previsão, mas não devem ser interpretadas sozinhas como justificativa para intervenção.

Outro ponto importante é que algumas relações descritivas podem parecer contraintuitivas quando vistas isoladamente. Isso acontece porque o modelo aprende interações entre múltiplas variáveis ao mesmo tempo, e não regras simples de uma coluna por vez.

### 7.6. Medidas práticas que podem ser tomadas com base no aprendizado

Com base no comportamento do modelo e na leitura dos dados, algumas ações fazem sentido:

- priorizar monitoramento de estudantes com `IPV`, `IPP`, `IDA`, `IEG`, `IAA` e `IPS` mais baixos
- reforçar acompanhamento pedagógico para estudantes com pior desempenho em matemática, português e inglês
- observar com atenção estudantes com menos avaliações registradas, porque isso pode indicar menor acompanhamento ou menor evidência sobre o progresso
- usar `fase` e `fase_ideal` apenas como contexto de priorização, e não como explicação principal do problema
- combinar o score do modelo com leitura humana da equipe pedagógica e psicossocial, evitando usar a previsão como decisão automática

### 7.7. Por que o baseline continua tão forte

Um ponto importante do projeto é que o `baseline_persistencia` continuou muito competitivo em relação aos modelos estatísticos. Em termos simples, esse baseline aplica a regra: se o estudante já está em defasagem hoje, há forte chance de continuar em risco no próximo ciclo. Como a defasagem apresenta persistência temporal relevante nesta base, essa regra simples já captura uma parte importante do comportamento observado.

Isso ajuda a explicar por que aumentar apenas o esforço de treinamento não tende, por si só, a resolver o problema. Os modelos comparados neste projeto não dependem de "mais eras" no sentido típico de redes neurais profundas. Em algoritmos como `random_forest` e `logistic_regression`, por exemplo, a limitação principal não é falta de iterações, mas a qualidade do sinal disponível nas variáveis de entrada. Em outras palavras: se os dados adicionais não trouxerem informação nova sobre mudança de trajetória, o modelo estatístico dificilmente ultrapassará uma regra simples que já aproveita um sinal muito forte do próprio problema.

Também é importante considerar que parte das variáveis mais fortes está muito próxima da própria definição de defasagem, como `fase`, `fase_ideal`, `IAN` e a defasagem atual. Essas variáveis podem ser úteis para previsão operacional, mas ajudam menos a descobrir causas acionáveis de melhora ou piora. Além disso, a base histórica útil para o alvo temporal ainda é relativamente curta, concentrada nas transições `2022 -> 2023` e `2023 -> 2024`, o que reduz a capacidade de o modelo aprender padrões mais finos de transição.

Assim, a conclusão não é que os modelos estatísticos não funcionam, mas que, com os dados atuais, eles ainda não encontraram sinal suficiente para superar o baseline no agregado. O ganho mais relevante apareceu em outra dimensão: transformar a leitura binária em uma visão mais rica de trajetória, distinguindo estudantes com chance de entrada em risco, persistência e recuperação.

O caminho mais promissor para evoluir essa solução não é simplesmente "treinar mais", mas sim:

- segmentar melhor o problema, separando estudantes hoje sem defasagem daqueles que já estão defasados
- criar variáveis temporais mais informativas, como variação de notas, engajamento e indicadores entre anos
- incorporar dados adicionais mais diretamente ligados à intervenção, como histórico de acompanhamento, frequência, participação e contexto de suporte
- tratar como objetivo central não apenas o risco binário futuro, mas também os movimentos de piora, estabilidade e recuperação

### 7.8. Conclusão final

O principal aprendizado do projeto é que a defasagem futura parece estar ligada a um conjunto articulado de sinais de desempenho, engajamento, acompanhamento e contexto escolar.

O modelo não substitui a análise pedagógica da equipe, mas funciona como um mecanismo de triagem e priorização. A camada binária ajuda a responder quem está em risco. A camada analógica complementar ajuda a responder quem parece estar subindo, descendo ou mantendo uma trajetória crítica.

Em outras palavras, a solução ajuda a responder:

> Quem merece atenção primeiro?

Esse é o ponto de maior valor da solução para a Passos Mágicos. O baseline forte não invalida a proposta: ele mostra que existe inércia importante no problema e que o maior valor dos modelos estatísticos, neste estágio, está em complementar a regra simples com priorização probabilística, leitura de exceções e apoio mais qualificado à intervenção.

---
