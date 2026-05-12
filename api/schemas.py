"""Pydantic v2 request/response models aligned with the German Credit Risk feature columns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreditRequest(BaseModel):
    """Single-observation scoring payload (features pre-processed to numeric codes)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    account_status: int = Field(
        ...,
        ge=1,
        le=4,
        description=(
            "Saldo da conta corrente: "
            "1=negativo; 2=0–200 DM; 3=>200 DM ou salário≥1 ano; 4=sem conta"
        ),
    )
    credit_duration: int = Field(
        ...,
        gt=0,
        le=120,
        description="Duração do crédito em meses (1–120).",
    )
    history_of_compliance: int = Field(
        ...,
        ge=0,
        le=4,
        description=(
            "Histórico de pagamentos: "
            "0=sem créditos/pontuais; 1=pontuais neste banco; 2=pontuais até hoje; "
            "3=atraso anterior; 4=conta problemática"
        ),
    )
    credit_purpose: int = Field(
        ...,
        ge=0,
        le=10,
        description=(
            "Finalidade: 0=carro novo; 1=carro usado; 2=móveis; 3=TV/rádio; "
            "4=eletrodomésticos; 5=reparos; 6=educação; 7=férias; "
            "8=reciclagem profissional; 9=negócios; 10=outros"
        ),
    )
    credit_amount: float = Field(
        ...,
        gt=0,
        description="Valor do crédito em DM (Deutschmark). Deve ser positivo.",
    )
    savings: int = Field(
        ...,
        ge=0,
        le=5,
        description=(
            "Saldo em poupança/investimentos: "
            "0=sem poupança; 1=<100 DM; 2=100–500 DM; 3=500–1000 DM; "
            "4=>1000 DM; 5=desconhecido"
        ),
    )
    employment_duration: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Tempo de emprego atual: " "1=desempregado; 2=<1 ano; 3=1–4 anos; 4=4–7 anos; 5=≥7 anos"
        ),
    )
    installment_rate: int = Field(
        ...,
        ge=1,
        le=4,
        description="Taxa de prestação em % da renda disponível (1=mais alta; 4=mais baixa).",
    )
    personal_status_sex: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Estado civil / sexo: "
            "1=homem divorciado; 2=mulher divorciada/casada; 3=homem solteiro; "
            "4=homem casado/viúvo; 5=mulher solteira"
        ),
    )
    other_debtors: int = Field(
        ...,
        ge=1,
        le=3,
        description="Outros devedores / garantias: 1=nenhum; 2=co-solicitante; 3=fiador.",
    )
    present_residence: int = Field(
        ...,
        ge=1,
        le=4,
        description="Tempo na residência atual em anos (1=<1; 2=1–4; 3=4–7; 4=≥7).",
    )
    property: int = Field(
        ...,
        ge=1,
        le=4,
        description=(
            "Bem mais valioso: "
            "1=imóvel; 2=poupança/seguro de vida; 3=carro ou outros; 4=desconhecido"
        ),
    )
    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Idade do tomador em anos (18–120).",
    )
    other_installment_plans: int = Field(
        ...,
        ge=1,
        le=3,
        description="Outros planos de parcelamento: 1=banco; 2=lojas; 3=nenhum.",
    )
    type_of_housing: int = Field(
        ...,
        ge=1,
        le=3,
        description="Tipo de moradia: 1=alugado; 2=próprio; 3=cedido/gratuito.",
    )
    number_credits: int = Field(
        ...,
        ge=1,
        le=4,
        description="Número de créditos existentes neste banco (1–4).",
    )
    job: int = Field(
        ...,
        ge=0,
        le=4,
        description=(
            "Ocupação: 0=desempregado/não-qualificado; 1=não-qualificado/residente; "
            "2=qualificado; 3=gestão/autônomo; 4=outro"
        ),
    )
    people_liable: int = Field(
        ...,
        ge=1,
        le=2,
        description="Número de dependentes financeiros: 1=≥3 pessoas; 2=0–2 pessoas.",
    )
    telephone: int = Field(
        ...,
        ge=-1,
        le=2,
        description="Telefone registrado: -1=não registrado; 1=sim (em nome do tomador); 2=outro.",
    )
    foreign_worker: int = Field(
        ...,
        ge=-1,
        le=1,
        description="Trabalhador estrangeiro: -1=sim; 0=informação ausente; 1=não.",
    )
    level_of_education: int = Field(
        ...,
        ge=-1,
        le=4,
        description="Nível de escolaridade (proxy): -1=sem dados; 0–4=crescente.",
    )
    entry_payment: float = Field(
        ...,
        ge=0.0,
        description="Valor de entrada / pagamento inicial em DM (≥ 0).",
    )

    @field_validator("credit_duration")
    @classmethod
    def duration_must_be_reasonable(cls, v: int) -> int:
        """Rejeita durações acima do limite prático para crédito ao consumidor."""
        if v > 120:
            raise ValueError("Duração não pode exceder 120 meses (10 anos).")
        return v

    @field_validator("credit_amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("O valor do crédito deve ser positivo.")
        return v


class CreditResponse(BaseModel):
    """Scoring output with probability, decision and risk category."""

    probability_good: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade de adimplência (credit_risk=1). Escala 0–1.",
    )
    probability_bad: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade de inadimplência (credit_risk=0). Escala 0–1.",
    )
    decision: Literal["aprovado", "em_analise", "negado"] = Field(
        ...,
        description=(
            "Decisão de crédito: "
            "'aprovado' (prob≥0.65); 'em_analise' (0.40≤prob<0.65); 'negado' (prob<0.40)"
        ),
    )
    risk_category: Literal["baixo", "medio", "alto"] = Field(
        ...,
        description="Categoria de risco derivada da probabilidade de inadimplência.",
    )
    threshold_used: float = Field(
        ...,
        description="Limiar de decisão utilizado nesta inferência.",
    )


class BatchScoringRequest(BaseModel):
    """Batch scoring payload — up to 100 records per call."""

    records: list[CreditRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de registros a pontuar (máximo 100 por chamada).",
    )


class BatchScoringResponse(BaseModel):
    """Batch scoring output."""

    results: list[CreditResponse]
    count: int = Field(..., description="Número de registros pontuados.")


class ModelInfo(BaseModel):
    """Model metadata for governance and auditability (BACEN 4.557)."""

    model_name: str
    version: str
    feature_count: int
    features: list[str]
    calibration_method: str
    status: Literal["loaded", "unavailable"]
