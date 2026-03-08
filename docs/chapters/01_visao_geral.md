## 1. Visão Geral

Esta solução foi desenvolvida para o Datathon da pós-graduação em Machine Learning Engineering da FIAP, com o case da Associação Passos Mágicos. Nesse datathon, o desafio foi construir todo o ciclo de vida de um modelo preditivo capaz de estimar o risco de defasagem escolar de cada estudante, aplicando as melhores práticas de MLOps, desde a construção do melhor modelo até o monitoramento contínuo em produção.

### Quem é a Passos Mágicos?

A Associação Passos Mágicos tem uma trajetória de 32 anos de atuação e trabalha na transformação da vida de crianças e jovens de baixa renda, os levando a melhores oportunidades de vida. A transformação, idealizada por Michelle Flues e Dimetri Ivanoff, começou em 1992, atuando dentro de orfanatos no município de Embu-Guaçu.

Em 2016, depois de anos de atuação, eles decidem ampliar o programa para que mais jovens tivessem acesso a essa fórmula mágica para transformação que inclui: educação de qualidade, auxílio psicológico/psicopedagógico, ampliação de sua visão de mundo e protagonismo. Passaram então a atuar como um projeto social e educacional, criando assim a Associação Passos Mágicos.

A associação busca instrumentalizar o uso da educação como ferramenta para a mudança das condições de vida das crianças e jovens em vulnerabilidade social.

O problema de negócio é estimar o `risco de defasagem escolar futura` de cada estudante. Em termos simples, a pergunta central é:

> Com os dados disponíveis sobre o estudante hoje, é possível antecipar quem tem maior chance de apresentar defasagem escolar no próximo ciclo?

Essa formulação foi tratada como um problema temporal:

- dados de `2022` são usados para prever risco em `2023`
- dados de `2023` são usados para prever risco em `2024`

Essa decisão torna o uso do modelo coerente com a operação real da instituição, porque o objetivo não é “reconstruir” uma situação já conhecida do mesmo ano, e sim antecipar risco para permitir intervenção mais cedo.

Na prática, a solução entrega:

- ingestão de arquivos CSV em layouts compatíveis com 2022, 2023 e 2024
- detecção automática do schema de entrada
- normalização para um schema canônico único
- consolidação histórica em PostgreSQL
- treino e comparação entre múltiplos modelos
- API FastAPI para operação
- dashboard visual para análise
- monitoramento com eventos, drift e relatórios exportáveis

### 1.1. O que acontece quando a API inicia

Quando a API é iniciada, ela executa apenas a preparação da aplicação. Em termos simples, ela:

1. sobe o servidor FastAPI
2. cria as pastas necessárias para logs, bundles e exportações
3. abre a conexão com o banco configurado
4. garante que as tabelas existam
5. publica os endpoints da API e as páginas do dashboard

O que ela **não** faz automaticamente ao iniciar:

- não carrega os CSVs para dentro do banco
- não treina os modelos
- não gera predições
- não gera relatórios de drift
- não recria artefatos antigos

Ou seja, a API inicia “pronta para operar”, mas o fluxo de dados e de Machine Learning só acontece quando os endpoints apropriados são chamados.

### 1.2. Explicação simples do que a solução faz

Em linguagem bem direta, a solução funciona como uma central de apoio à decisão da Passos Mágicos.

Ela recebe arquivos com dados dos estudantes, organiza tudo em um padrão único, grava isso no banco, treina modelos com o histórico disponível e depois responde perguntas como:

- quais estudantes têm maior chance de entrar em defasagem no próximo ciclo
- quais estudantes já estão em risco e tendem a continuar assim
- quais estudantes têm chance de recuperação
- se os dados novos estão diferentes demais da base usada no treino

O objetivo é ajudar a priorizar atenção, triagem e acompanhamento.

Base configurada para publicação:

- `https://rodrigojager.com/datathon`

---
