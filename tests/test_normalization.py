from src.normalization import normalize_uploaded_csv


def test_normalizes_2022_columns_to_canonical_schema():
    content = (
        "RA,Fase,Turma,Idade 22,Gênero,Ano ingresso,Instituição de ensino,INDE 22,IAA,IEG,IPS,IDA,Matem,Portug,Inglês,IPV,IAN,Fase ideal,Defas\n"
        "RA-1,7,A,19,Menina,2016,Escola Pública,\"5,783\",\"8,3\",\"4,1\",\"5,6\",\"4,0\",\"2,7\",\"3,5\",\"6,0\",\"7,278\",\"5,000\",Fase 8,-1\n"
    ).encode("utf-8")

    result = normalize_uploaded_csv(
        content=content,
        filename="PEDE2022_teste.csv",
    )

    row = result.dataframe.iloc[0]
    assert result.detected_schema == "pede_2022_like"
    assert result.reference_year == 2022
    assert row["ra"] == "RA-1"
    assert row["idade"] == 19
    assert row["nota_matematica"] == 2.7
    assert row["defasagem"] == -1


def test_normalizes_2024_like_status_and_extra_columns():
    content = (
        "RA,Fase,INDE 2024,Turma,Idade,Gênero,Ano ingresso,Instituição de ensino,IPP,Mat,Por,Ing,IAN,Fase Ideal,Defasagem,Escola,Ativo/ Inativo\n"
        "RA-10,ALFA,\"7,61\",ALFA A,8,Masculino,2024,Pública,\"5,6\",\"10,0\",\"6,0\",,\"7,1\",ALFA,0,EE X,Cursando\n"
    ).encode("utf-8")

    result = normalize_uploaded_csv(content=content, filename="arquivo_teste_2024.csv")

    row = result.dataframe.iloc[0]
    assert result.detected_schema == "pede_2024_like"
    assert bool(row["status_matricula"]) is True
    assert row["ipp"] == 5.6
    assert row["inde_atual"] == 7.61
