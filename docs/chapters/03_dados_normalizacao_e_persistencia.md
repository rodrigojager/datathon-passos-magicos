## 3. Dados, Normalização e Persistência

### 3.1. Fontes de dados

Os dados do case estão distribuídos em três arquivos:

- `BASE DE DADOS - DATATHON - PEDE2022.csv`
- `BASE DE DADOS - DATATHON - PEDE2023.csv`
- `BASE DE DADOS - DATATHON - PEDE2024.csv`

Cada linha representa um estudante em um ano específico.

Exemplos de variáveis:

- `RA`
- `Fase`
- `Turma`
- `Idade`
- `Gênero`
- `Ano ingresso`
- `Instituição de ensino`
- `INDE`
- `IAA`
- `IEG`
- `IPS`
- `IPP`
- `IDA`
- `nota de matemática`
- `nota de português`
- `nota de inglês`
- `IPV`
- `IAN`
- `Fase ideal`
- `Defasagem`

### 3.2. Normalização dos arquivos

Todos os endpoints que recebem arquivo usam a mesma rotina de normalização.

Etapas executadas:

1. leitura tolerante a separadores e encoding
2. padronização técnica dos nomes das colunas
3. resolução de aliases
4. detecção do layout compatível
5. montagem do schema canônico
6. coerção de tipos numéricos e booleanos

Exemplos de aliases tratados:

- `Matem` e `Mat` viram `nota_matematica`
- `Portug` e `Por` viram `nota_portugues`
- `Defas` e `Defasagem` viram `defasagem`
- `Idade 22` e `Idade` viram `idade`

### 3.3. Deduplicação e consolidação

Na ingestão, a chave lógica adotada é:

- `ra + ano_referencia`

Se essa combinação já existir no banco, o registro não é inserido novamente.

A base consolidada final é persistida na tabela:

- `student_records`

### 3.4. Tabelas principais do PostgreSQL

- `student_records`: base histórica consolidada por aluno e ano
- `ingestion_runs`: histórico de ingestões
- `training_runs`: histórico de treinos e candidatos comparados
- `training_reference_rows`: base de referência do último treino para drift
- `prediction_runs` e `prediction_rows`: execuções de predição
- `evaluation_runs` e `evaluation_rows`: comparações previsto x real e comparações entre modelos
- `drift_runs`: resultados de drift, inclusive o HTML do relatório
- `app_events`: eventos operacionais usados no dashboard de logs

### 3.5. Versionamento de dados

Cada treino, avaliação ou análise de drift registra a versão da base usada na execução.

Formato:

- `v{ultima_ingestao}-r{quantidade_de_linhas}`

Exemplo:

- `v3-r3030`

---
