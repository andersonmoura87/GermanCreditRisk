"""Streamlit demo — German Credit Risk Scoring.

Demo interativo para stakeholders: analistas de crédito, gestores de produto e auditores.

Como executar:
    poetry run streamlit run streamlit_app.py
    # ou
    make streamlit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="German Credit Risk — Score",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS mínimo para badges coloridos
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .badge-aprovado  { background:#639922; color:#fff; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .badge-analise   { background:#f5c842; color:#333; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .badge-negado    { background:#E24B4A; color:#fff; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .metric-box      { background:#f7f7f7; border-radius:8px; padding:12px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Mapeamentos legíveis para o analista
# ---------------------------------------------------------------------------
ACCOUNT_STATUS = {1: "Saldo negativo (< 0 DM)", 2: "0–200 DM", 3: "> 200 DM / salário", 4: "Sem conta corrente"}
COMPLIANCE     = {0: "Sem histórico / pago pontualmente", 1: "Pago neste banco", 2: "Pago até hoje",
                  3: "Atraso anterior", 4: "Conta problemática"}
PURPOSE        = {0: "Carro novo", 1: "Carro usado", 2: "Móveis", 3: "TV / Rádio",
                  4: "Eletrodomésticos", 5: "Reparos", 6: "Educação", 7: "Férias",
                  8: "Reciclagem profissional", 9: "Negócios", 10: "Outros"}
SAVINGS        = {0: "Sem poupança", 1: "< 100 DM", 2: "100–500 DM", 3: "500–1000 DM",
                  4: "> 1000 DM", 5: "Desconhecido"}
EMPLOYMENT     = {1: "Desempregado", 2: "< 1 ano", 3: "1–4 anos", 4: "4–7 anos", 5: "≥ 7 anos"}
PERS_STATUS    = {1: "Homem divorciado", 2: "Mulher divorciada/casada", 3: "Homem solteiro",
                  4: "Homem casado/viúvo", 5: "Mulher solteira"}
DEBTORS        = {1: "Nenhum", 2: "Co-solicitante", 3: "Fiador"}
PROPERTY       = {1: "Imóvel", 2: "Poupança / seguro de vida", 3: "Carro ou outros", 4: "Desconhecido"}
INSTALLMENTS   = {1: "Banco", 2: "Lojas", 3: "Nenhum"}
HOUSING        = {1: "Alugado", 2: "Próprio", 3: "Cedido / gratuito"}
JOB            = {0: "Desempregado / não-qualificado", 1: "Não-qualificado / residente",
                  2: "Qualificado", 3: "Gestão / autônomo", 4: "Outro"}
TELEPHONE      = {-1: "Não registrado", 1: "Sim (em nome do tomador)", 2: "Outro"}
FOREIGN        = {-1: "Sim", 0: "Sem informação", 1: "Não"}
EDUCATION      = {-1: "Sem dados", 0: "Básica incompleta", 1: "Básica", 2: "Médio",
                  3: "Superior incompleto", 4: "Superior completo"}


def _select(label: str, mapping: dict[int, str], default_key: int) -> int:
    keys = list(mapping.keys())
    labels = [mapping[k] for k in keys]
    idx = keys.index(default_key) if default_key in keys else 0
    sel = st.selectbox(label, options=labels, index=idx)
    return keys[labels.index(sel)]


# ---------------------------------------------------------------------------
# Sidebar — formulário de entrada
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💳 Perfil do Tomador")
    st.caption("Preencha os dados para simular a análise de crédito.")

    st.subheader("Conta e histórico")
    account_status        = _select("Status da conta corrente", ACCOUNT_STATUS, 1)
    history_of_compliance = _select("Histórico de pagamentos", COMPLIANCE, 2)
    savings               = _select("Poupança / investimentos", SAVINGS, 1)

    st.subheader("Crédito solicitado")
    credit_duration = st.slider("Prazo (meses)", 1, 120, 24)
    credit_amount   = st.number_input("Valor (DM)", min_value=100.0, max_value=20_000.0,
                                       value=3_000.0, step=100.0)
    credit_purpose  = _select("Finalidade", PURPOSE, 0)
    installment_rate = st.selectbox("Taxa de prestação (% renda)",
                                    options=[1, 2, 3, 4],
                                    format_func=lambda x: f"{x} — {'alta' if x==1 else 'baixa' if x==4 else 'média'}",
                                    index=1)
    entry_payment   = st.number_input("Entrada / pagamento inicial (DM)", min_value=0.0,
                                       max_value=float(credit_amount), value=0.0, step=100.0)

    st.subheader("Perfil pessoal")
    age               = st.slider("Idade (anos)", 18, 80, 35)
    personal_status_sex = _select("Estado civil / sexo", PERS_STATUS, 3)
    employment_duration = _select("Tempo no emprego atual", EMPLOYMENT, 3)
    job               = _select("Ocupação", JOB, 2)
    level_of_education = _select("Nível de escolaridade", EDUCATION, 2)
    telephone         = _select("Telefone registrado", TELEPHONE, 1)
    foreign_worker    = _select("Trabalhador estrangeiro", FOREIGN, 1)

    st.subheader("Garantias e moradia")
    property_val        = _select("Bem mais valioso", PROPERTY, 2)
    type_of_housing     = _select("Tipo de moradia", HOUSING, 1)
    present_residence   = st.selectbox("Tempo na residência (anos)",
                                        options=[1, 2, 3, 4],
                                        format_func=lambda x: {1: "<1", 2: "1–4", 3: "4–7", 4: "≥7"}[x],
                                        index=1)
    other_debtors       = _select("Outros devedores", DEBTORS, 1)

    st.subheader("Obrigações financeiras")
    number_credits       = st.selectbox("Créditos neste banco", [1, 2, 3, 4], index=0)
    other_installment_plans = _select("Outros parcelamentos", INSTALLMENTS, 3)
    people_liable        = st.selectbox("Dependentes financeiros",
                                         options=[1, 2],
                                         format_func=lambda x: "≥ 3 pessoas" if x == 1 else "0–2 pessoas",
                                         index=1)

    st.divider()
    run_score = st.button("▶ Analisar crédito", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------
st.title("German Credit Risk — Análise de Crédito")
st.caption(
    "Demo de score de crédito baseado no dataset German Credit Risk (GCR). "
    "Modelo calibrado com Platt scaling · análise de fairness integrada."
)

if not run_score:
    st.info(
        "Preencha o perfil do tomador na barra lateral e clique em **Analisar crédito**.",
        icon="👈",
    )
    with st.expander("ℹ️ Sobre este modelo"):
        st.markdown(
            """
            **Metodologia**
            - Regressão logística calibrada (Platt scaling) treinada no GCR
            - Features codificadas via WoE (Weight of Evidence)
            - Métricas de mercado: KS statistic, Gini coefficient, AUC-ROC

            **Decisão**
            | Score | Decisão | Risco |
            |-------|---------|-------|
            | ≥ 65% | Aprovado | Baixo |
            | 40–65% | Em análise | Médio |
            | < 40% | Negado | Alto |

            **Assimetria de custo:** o custo de conceder crédito a um mau pagador
            é 5× maior do que recusar a um bom pagador (BACEN Res. 4.557).

            **Fairness & LGPD:** o modelo monitora viés por gênero e faixa etária.
            Nenhuma variável sensível é usada diretamente na decisão.
            """
        )
    st.stop()

# ---------------------------------------------------------------------------
# Executa o scoring
# ---------------------------------------------------------------------------
payload: dict = {
    "account_status": account_status,
    "credit_duration": credit_duration,
    "history_of_compliance": history_of_compliance,
    "credit_purpose": credit_purpose,
    "credit_amount": credit_amount,
    "savings": savings,
    "employment_duration": employment_duration,
    "installment_rate": installment_rate,
    "personal_status_sex": personal_status_sex,
    "other_debtors": other_debtors,
    "present_residence": present_residence,
    "property": property_val,
    "age": age,
    "other_installment_plans": other_installment_plans,
    "type_of_housing": type_of_housing,
    "number_credits": number_credits,
    "job": job,
    "people_liable": people_liable,
    "telephone": telephone,
    "foreign_worker": foreign_worker,
    "level_of_education": level_of_education,
    "entry_payment": entry_payment,
}

try:
    from src.models.predict import predict_proba

    row = pd.DataFrame([payload])
    prob_good = float(predict_proba(row)[0])
    prob_bad  = round(1.0 - prob_good, 6)

    if prob_good >= 0.65:
        decision, risk_cat, badge_cls, bar_color = "Aprovado", "Baixo", "badge-aprovado", "#639922"
    elif prob_good >= 0.40:
        decision, risk_cat, badge_cls, bar_color = "Em análise", "Médio", "badge-analise", "#f5c842"
    else:
        decision, risk_cat, badge_cls, bar_color = "Negado", "Alto", "badge-negado", "#E24B4A"

    model_ok = True

except FileNotFoundError:
    model_ok = False
    st.error(
        "**Modelo não encontrado.** Execute `make train` e reinicie o app.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Layout de resultado — 3 colunas
# ---------------------------------------------------------------------------
col_result, col_details, col_features = st.columns([1.2, 1.2, 1.6], gap="large")

with col_result:
    st.subheader("Resultado")
    st.markdown(f'<span class="{badge_cls}">{decision}</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Gauge como progress bar customizado
    st.markdown(f"**Probabilidade de adimplência**")
    st.progress(prob_good, text=f"{prob_good:.1%}")

    st.metric("Risco", risk_cat)
    st.metric("Prob. inadimplência", f"{prob_bad:.1%}")

with col_details:
    st.subheader("Dados do pedido")
    st.markdown(
        f"""
        | Campo | Valor |
        |---|---|
        | Valor solicitado | DM {credit_amount:,.0f} |
        | Prazo | {credit_duration} meses |
        | Finalidade | {PURPOSE[credit_purpose]} |
        | Idade | {age} anos |
        | Emprego | {EMPLOYMENT[employment_duration]} |
        | Conta corrente | {ACCOUNT_STATUS[account_status]} |
        | Poupança | {SAVINGS[savings]} |
        | Moradia | {HOUSING[type_of_housing]} |
        """
    )

    st.markdown("---")
    st.caption(
        "⚖️ **Nota de fairness (LGPD):** o score não utiliza diretamente "
        "gênero, etnia ou religião como variáveis preditoras. "
        "Viés por faixa etária e gênero é monitorado continuamente."
    )

with col_features:
    st.subheader("Poder preditivo das features (IV)")

    iv_path = PROJECT_ROOT / "data" / "processed" / "iv_summary.csv"
    if iv_path.is_file():
        import matplotlib.pyplot as plt

        iv_df = pd.read_csv(iv_path).nlargest(12, "iv").reset_index(drop=True)

        # Paleta por faixa de IV
        def _iv_color(iv: float) -> str:
            if iv < 0.02: return "#aaaaaa"
            if iv < 0.1:  return "#f5c842"
            if iv < 0.3:  return "#639922"
            if iv < 0.5:  return "#2e86ab"
            return "#E24B4A"

        colors = [_iv_color(v) for v in iv_df["iv"]]
        fig, ax = plt.subplots(figsize=(5, max(3, len(iv_df) * 0.38)))
        ax.barh(iv_df["feature"][::-1], iv_df["iv"][::-1], color=colors[::-1])
        ax.axvline(0.1, color="#aaa", linestyle="--", linewidth=0.7)
        ax.axvline(0.3, color="#639922", linestyle="--", linewidth=0.7)
        ax.set_xlabel("Information Value")
        ax.set_title("Top features por IV")
        fig.tight_layout()
        st.pyplot(fig)

        st.caption(
            "IV < 0.02 = inútil · 0.02–0.1 = fraco · 0.1–0.3 = médio · > 0.3 = forte"
        )
    else:
        st.info(
            "Execute `make train` para gerar `data/processed/iv_summary.csv` "
            "e visualizar o ranking de features aqui.",
            icon="📊",
        )

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "German Credit Risk API · Modelo calibrado (Platt scaling) · "
    "Governança BACEN 4.557/2017 · LGPD · "
    "[Documentação da API](http://localhost:8000/docs)"
)
