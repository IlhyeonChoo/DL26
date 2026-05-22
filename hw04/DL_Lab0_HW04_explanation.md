# DL Lab HW04 Implementation Notes

This directory implements `Todo.md` as a reproducible Jupyter notebook.

## Files

- `DL_Lab0_HW04.ipynb`: main experiment notebook.
- `build_hw04_notebook.py`: deterministic notebook generator.
- `Todo.md`: original assignment prompt.

## Implemented Tasks

### Problem 1: RNN / LSTM / GRU

The notebook trains and compares three recurrent forecasting models on ETTh1:

- RNN
- LSTM
- GRU

The fixed setting is:

- input shape: `(batch, 96, 7)`
- output shape: `(batch, 24)`
- hidden dimension: `64`
- recurrent layers: `1`
- optimizer: `AdamW`
- metrics: MSE and MAE

The notebook records:

- parameter counts
- train and validation MSE curves
- test MSE and MAE
- forecast visualization for one test interval

### Problem 2: Input Length and Bidirectional Comparison

The notebook selects the best recurrent model from Problem 1 by test MSE, then compares:

- input lengths: `24`, `48`, `96`, `192`
- bidirectional: `False` vs `True`

The notebook records:

- test MSE and MAE by input length
- test MSE and MAE before and after bidirectional recurrent encoding
- analysis prompts for interpreting causality and sequence-length trade-offs

### Problem 3: TransformerEncoder Forecasting

The notebook implements:

```text
Linear input projection + learnable positional embedding
-> nn.TransformerEncoder
-> fully connected prediction head
```

The default Transformer setting is:

- `d_model = 64`
- `nhead = 4`
- `num_layers = 2`
- `dim_feedforward = 128`
- learnable positional encoding via `nn.Parameter`

The notebook records:

- parameter count
- train and validation MSE curves
- test MSE and MAE
- comparison with the best Problem 1 recurrent model
- forecast visualization for one test interval

## Running

From the repository root:

```bash
python3 hw04/build_hw04_notebook.py
```

Then open and run:

```text
hw04/DL_Lab0_HW04.ipynb
```

The notebook downloads ETTh1 automatically from the public ETT dataset repository if `hw04/data/ETTh1.csv` is not already present.

Results are saved under:

```text
hw04/results_hw04/
```

On Colab, the notebook saves to:

```text
/content/drive/MyDrive/DL26/hw04/results_hw04
```

if Google Drive mounting succeeds.
