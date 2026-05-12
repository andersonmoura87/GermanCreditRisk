# German Credit Risk: Uma Análise Detalhada

O German Credit Risk (GCR) é um conjunto de dados fascinante que oferece uma visão profunda do cenário de crédito na Alemanha. Reunindo informações de mais de 1.000 tomadores de crédito, o dataset permite explorar diversos aspectos do crédito e da inadimplência.

## O que o GCR contém:

- **Dados Demográficos:** Idade, sexo, estado civil e renda dos tomadores de crédito.
- **Histórico de Crédito:** Informações sobre pagamentos em atraso, inadimplência e outras pendências.
- **Características do Empréstimo:** Valor do empréstimo, prazo, taxa de juros e tipo de garantia.
- **Situação Profissional:** Tipo de emprego, renda e tempo de serviço.
- **Outros Fatores:** Moradia, estado civil, educação e histórico de moradia.

## O que você pode fazer com o GCR:

- **Analisar o perfil de risco de tomadores de crédito:** Identificar características que aumentam ou diminuem o risco de inadimplência.
- **Desenvolver modelos de previsão de crédito:** Criar modelos para prever a probabilidade de um tomador de crédito inadimplir.
- **Avaliar a qualidade de carteiras de crédito:** Analisar o risco de uma carteira de crédito e identificar os tomadores de crédito mais arriscados.
- **Estudar os fatores que influenciam o risco de crédito:** Investigar como fatores como idade, renda e histórico de crédito afetam o risco de inadimplência.
- **Comparar diferentes modelos de análise de crédito:** Avaliar o desempenho de diferentes modelos e escolher o mais adequado para cada situação.

## O GCR é um recurso valioso para:

- **Profissionais de crédito:** Bancos, instituições financeiras e empresas de análise de crédito.
- **Pesquisadores:** Acadêmicos e estudantes que estudam economia, finanças ou risco de crédito.
- **Entusiastas de dados:** Qualquer pessoa interessada em aprender sobre análise de dados e machine learning.

## Para acessar o GCR:

- **Download:** O dataset está disponível gratuitamente para download em diversos repositórios online.
- **Ferramentas:** Existem diversas ferramentas online e offline que permitem analisar o GCR.
- **Tutoriais:** Há uma variedade de tutoriais e cursos online que ensinam como usar o GCR.

Explore o GCR e descubra as histórias que ele conta!

*Lembre-se: O GCR é um dataset real e contém informações confidenciais. É importante usar o dataset de forma responsável e ética.*

*Link para download:* https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29

*Contém a documentação e um breve dicionário de dados referente ao dataset.*

---

## Fonte e citação do dataset

**Autor:** Ulrike Grömping — Beuth University of Applied Sciences Berlin  
**Contato:** https://prof.beuth-hochschule.de/groemping/

> Este repositório utiliza a versão corrigida **South German Credit Data** (Grömping, 2019), que corrige erros de codificação da versão Statlog original (novembro/2019) com base em Open Data LMU (2010) e outras fontes em alemão.

### Dicionário de variáveis

| Variável | Tipo | Descrição |
|---|---|---|
| `account_status` | cat. | Status da conta corrente: 1=sem conta, 2=<0 DM, 3=0–200 DM, 4=≥200 DM |
| `credit_duration` | quant. | Duração do crédito em meses |
| `history_of_compliance` | cat. | Histórico de cumprimento: 0=atraso, 1=crítico, 2=quitado, 3=em dia, 4=tudo quitado no banco |
| `credit_purpose` | cat. | Finalidade: 0=outros, 1=carro novo, 2=carro usado, 3=móveis, 4=rádio/TV, 5=eletrodom., 6=reparos, 7=educação, 8=férias, 9=requalificação, 10=negócios |
| `credit_amount` | quant. | Valor do crédito em DM (transformação monotônica; dados originais desconhecidos) |
| `savings` | cat. | Poupança: 1=desconhecido/sem conta, 2=<100 DM, 3=100–500 DM, 4=500–1000 DM, 5=≥1000 DM |
| `employment_duration` | ord. | Tempo no emprego atual: 1=desempregado, 2=<1 ano, 3=1–4 anos, 4=4–7 anos, 5=≥7 anos |
| `installment_rate` | ord. | Prestação como % da renda disponível: 1=≥35%, 2=25–35%, 3=20–25%, 4=<20% |
| `personal_status_sex` | cat. | Sexo/estado civil: 1=M divorciado, 2=F não-solteira ou M solteiro, 3=M casado/viúvo, 4=F solteira |
| `other_debtors` | cat. | Outros devedores/garantidores: 1=nenhum, 2=co-solicitante, 3=fiador |
| `present_residence` | ord. | Tempo na residência atual: 1=<1 ano, 2=1–4 anos, 3=4–7 anos, 4=≥7 anos |
| `property` | ord. | Bem mais valioso: 1=nenhum, 2=carro/outros, 3=seguro de vida, 4=imóvel |
| `age` | quant. | Idade em anos |
| `other_installment_plans` | cat. | Planos de parcelamento externos: 1=banco, 2=lojas, 3=nenhum |
| `type_of_housing` | cat. | Moradia: 1=gratuita, 2=aluguel, 3=própria |
| `number_credits` | ord. | Número de créditos no banco: 1=1, 2=2–3, 3=4–5, 4=≥6 |
| `job` | ord. | Qualificação profissional: 1=desempregado/não-residente, 2=não-qualificado, 3=qualificado, 4=gerente/autônomo |
| `people_liable` | bin. | Dependentes financeiros: 1=3 ou mais, 2=0–2 |
| `telephone` | bin. | Telefone fixo registrado no nome |
| `level_of_education` | cat. | Nível de escolaridade |
| `entry_payment` | quant. | Entrada em DM (transformação monotônica) |
| `credit_risk` | bin. | **Target** — 0=mau pagador (inadimplente), 1=bom pagador (adimplente) |

### Referências

- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set.* Report 4/2019, Beuth University of Applied Sciences Berlin.
- Fahrmeir, L. & Hamerle, A. (1984). *Multivariate Statistische Verfahren* (1st ed., Ch. 8). De Gruyter, Berlin.
- Open Data LMU (2010). *Kreditscoring zur Klassifikation von Kreditnehmern.* Accessed Nov 27, 2019.

---

## Projeto sênior (estrutura de código)

Este repositório segue a estrutura e o checklist completo (identidade, stack, macro BR, notebooks, domínio) em [`AGENTS.md`](AGENTS.md). O CSV bruto do GCR fica em **`data/raw/credit_risk_dataset.csv`** (não editar).

**Pré-requisitos:** Python 3.11–3.13 e [Poetry](https://python-poetry.org/).

Instalação **rápida** (pipeline baseline + API + testes): `poetry install`.

Para notebooks “sênior” (SHAP, ydata-profiling, XGBoost/LightGBM, Evidently, etc.): `poetry install --with full`.

Testes sem medir cobertura (mais rápido): `poetry run pytest --no-cov -q`.

**CI/CD & segurança (GitHub Actions):** jobs paralelos `lint`, `test` e `security` (Bandit + pip-audit); em **PRs**, *Dependency Review* (vulnerabilidades em dependências). Em **PRs/branch:** `make test-ci`; no **push** `main`/`master`: cobertura + testes lentos (GX) + **Trivy** na imagem Docker e upload de SARIF (Code scanning, quando disponível). O build Docker exige `lint`, `test` e `security` OK. **Dependabot** em `.github/dependabot.yml` mantém Actions e `poetry.lock` atualizados.

```text
poetry install
poetry run python -m src.data.make_dataset
poetry run python -m src.data.validate
poetry run python -m src.features.build_features
poetry run python -m src.models.train
```

Ou, com [DVC](https://dvc.org/) inicializado (`dvc init`): `make train`.

**API local:** `make serve` (ou `poetry run uvicorn api.main:app --reload --port 8000`). É necessário o artefato `models/baseline.joblib` após o treino.

O CSV bruto deve estar em `data/raw/credit_risk_dataset.csv`.

