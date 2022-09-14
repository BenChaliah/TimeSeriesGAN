[![Maintenance](https://img.shields.io/badge/Maintained-yes-green.svg)]() [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)

## Time series analysis using GANs

The model in this repository combines a stack of dilated causal convolutional and LSTM layers, and while GANs are quite challenging to train, this approach have produced great results on various types of datasets, such as stock market and weather measurements.

<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/DCC.png?raw=true" alt="dilated causal convolutional layers"/>
</p>

```
├── data
│   └── sample.csv
├── databricks-ci
│   ├── create_cls.sh
│   ├── runNotebook.sh
│   └── start_cls.sh
├── .gitlab-ci.yml
├── Standalone_example.ipynb
└── TSGAN_Hypertuning.ipynb
```
### Requirements
- Versions are just recommendation
```
Databricks ML runtime (e.g. 10.5.x-cpu-ml)
Optuna 3.0.1
MLflow 1.8.0
torch 1.5.0
torchvision 0.6.0
```

### Data extraction: 
- The following script uses Spark for fast extraction of the latest financial data from Yahoo Finance API

```python
tickers = ["GOOG", "AAPL"]
df = spark.createDataFrame([(i,) for i in tickers],("ticker",))

return_type = ArrayType(MapType(StringType(), StringType()))

@udf(returnType=return_type)
def yahoo_udf(ticker_label):

    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:74.0) Gecko/20100101 Firefox/74.0'}
    end_ts = int(time.time())
    start_ts = end_ts - 3600*24*7
    ticker = ticker_label
    data_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?symbol={ticker}&period1={start_ts}&period2={end_ts}&useYfid=true&interval=1m&includePrePost=true"
    values = None
    data_dict = {i: [] for i in ['timestamp', 'low', 'high', 'close', 'open', 'volume']}
    try: 
        f = urllib.request.urlopen(urllib.request.Request(data_url, headers=headers))
        status = f.status
        if status == 200:
            jresp = json.loads(f.read().decode("utf-8"))    
            data_dict['timestamp'] = jresp['chart']['result'][0]['timestamp']
            for k in ['low', 'high', 'close', 'open', 'volume']:
                data_dict[k] = jresp['chart']['result'][0]['indicators']['quote'][0][k]
    except:
        pass
    df = pd.DataFrame(data_dict)
    values = df.to_dict("index").values()
    return list(values)

extracted = yahoo_udf("ticker")
exploded = explode(extracted).alias("exploded")
expanded = [
    col("exploded").getItem(k).alias(k) for k in ['timestamp', 'low', 'high', 'close', 'open', 'volume']
]

pdf = df.select("ticker", exploded).select("ticker", *expanded)
```

### Hyperparameter optimization
- Optuna and Mlflow were through case studies and expirements used to provide the optimal choices for learning rate, optimization algorithm, dropout ratio for the Generator and the Discriminator independently 
<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/exp_2.png?raw=true" alt="" width="100%"/>
</p>

```python
def suggest_hyperparameters(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    dropoutG = trial.suggest_float("dropoutG", 0.0, 0.4, step=0.1)
    dropoutD = trial.suggest_float("dropoutD", 0.0, 0.4, step=0.1)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

    return lr, dropoutG, dropoutD, optimizer_name

def objective(trial):
    best_val_loss = float('Inf')
    nz_dim = 100
    best_mse_val = None
    ...
    with mlflow.start_run():

        lr, dropoutG, dropoutD, optimizer_name = suggest_hyperparameters(trial)
        ...
        mlflow.log_params(trial.params)
        mlflow.log_param("device", device)

        netG = Generator(in_dim=nz_dim, n_channel=10, kernel_size=8, out_dim=1, hidden_dim=100, dropout=dropoutG).to(device)
        netD = Discriminator(in_dim=1, cnn_layers=4, n_layers=1, kernel_size=8, n_channel=10, dropout=dropoutD, hidden_dim=100).to(device)

        optimizerD = getattr(optim, optimizer_name)(netD.parameters(), lr=lr)
        optimizerG = getattr(optim, optimizer_name)(netG.parameters(), lr=lr)
        
        train(netG, netD, device, train_dataloader, optimizerG, optimizerD, n_epochs)
        mse_errG = test(netG, device, test_dataloader)
        
        if best_mse_val is None:
            best_mse_val = mse_errG
        best_mse_val = min(best_mse_val, mse_errG)
        mlflow.log_metric("mse_errG", mse_errG)

    return best_mse_val


run_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
experiment_id = mlflow.create_experiment(
    f"/Users/user/TimeSeriesGAN_Exp_{run_tag}",
    tags={"version": "v1", "priority": "P1"},
)

mlflow.set_experiment(experiment_id=experiment_id)
study = optuna.create_study(study_name=f"TimeSeriesGAN_study_{run_tag}", direction="minimize")
study.optimize(objective, n_trials=10)
```

### Results
 > `sample.csv`
- D(G(z))
<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/D_G_z.png?raw=true" alt="" width="75%"/>
</p>

__Evolution of the generated time series__
- Epoch 67
<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/fake_step_66.png?raw=true" alt="synthetic time series" width="75%"/>
</p>

- Epoch 135
<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/fake_step_134.png?raw=true" alt="synthetic time series" width="75%"/>
</p>

- Epoch 200
<p align="center">
  <img src="https://github.com/BenChaliah/TimeSeriesGAN/blob/main/assets/fake_step_199.png?raw=true" alt="synthetic time series" width="75%"/>
</p>
