STRATEGY_A_FEATURES = [
    "fase",
    "turma",
    "idade",
    "genero",
    "ano_ingresso",
    "instituicao_ensino",
    "numero_avaliacoes",
    "inde_atual",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "nota_matematica",
    "nota_portugues",
    "nota_ingles",
    "ipv",
]

STRATEGY_B_FEATURES = STRATEGY_A_FEATURES + [
    "ian",
    "fase_ideal",
    "ipp",
    "flag_ipp_ausente",
]

CATEGORICAL_FEATURES = {
    "fase",
    "turma",
    "genero",
    "instituicao_ensino",
    "fase_ideal",
}
