# Data Directory

Este diretório separa dados brutos e dados processados usados pelos experimentos.

## `data/raw/`

Armazena arquivos baixados diretamente da fonte original ou muito próximos dela. No estado atual do repositório, o uso mais direto é:

- `main_prices.csv`, gerado por `experiments/run_price_data_setup.py`

Esses dados podem ser obtidos programaticamente via `yfinance`, conforme implementado em `src/crypto_price_forecasting/data.py`.

## `data/processed/`

Armazena versões derivadas, filtradas, sincronizadas ou transformadas dos dados brutos, caso necessário em etapas futuras do trabalho. O objetivo é manter separado o que foi baixado da fonte do que foi produzido internamente no pipeline.

## Reprodutibilidade

- Os dados financeiros dependem de datas de início e fim, janelas de treino/teste e ativos selecionados.
- Para reprodutibilidade, registre sempre:
  - intervalo temporal utilizado
  - lista de ativos
  - data em que o download foi executado
- O repositório já fixa janelas centrais em `src/crypto_price_forecasting/config.py`, mas os scripts também aceitam parâmetros explícitos.

## Versionamento

- Arquivos brutos grandes não precisam ser versionados no Git.
- Prefira versionar apenas arquivos pequenos, metadados ou amostras mínimas quando necessário para auditoria.
- Se um conjunto bruto for regenerável por script, a prática recomendada é documentar o procedimento e não commitar o arquivo completo.
