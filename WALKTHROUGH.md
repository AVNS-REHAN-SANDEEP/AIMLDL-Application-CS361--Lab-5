# CS361 -- Lab Assignments 4 & 5: Full Walkthrough

## 0. What's in this bundle

| File | Purpose |
|---|---|
| `lab4_cnn_experiments.py` | Rebuilds the original overfitting CNN + 4 fixes + a combined best model, trains all of them, saves plots/metrics, saves `best_model.keras` |
| `lab5_streamlit_app.py` | Web app (Streamlit) that loads `best_model.keras` and classifies an uploaded image |
| `lab5_flask_app.py` | Same app, Flask version, single file, no template folder needed |
| `WALKTHROUGH.md` | This document -- setup steps + the analysis Lab 4 asks for |

Both lab scripts are plain, runnable Python -- no notebook required, though you can paste each function into notebook cells if your instructor wants a `.ipynb`.

---

## 1. Setup (do this once)

```bash
pip install tensorflow matplotlib pillow streamlit flask gdown
```

Get the dataset (Colab or any machine with internet access -- this step needs Google Drive, which isn't reachable from every sandboxed environment):

```python
import gdown
gdown.download(id="1XdXz0TKo_KCDRHOMvzV-YtcTx7NPG-jC", output="clothing_dataset.zip", quiet=False)
```
```bash
unzip -q clothing_dataset.zip -d clothing_dataset
```

Check the folder structure it produces:
```python
import os
print(os.listdir("clothing_dataset"))
```
Point `DATA_DIR` at the top of `lab4_cnn_experiments.py` to wherever the 10 class-named subfolders end up (the script auto-handles either a `train/`+`test/` layout or a single flat folder of class subfolders -- see the docstring at the top of the file).

---

## 2. Running Lab 4

```bash
python lab4_cnn_experiments.py
```

This trains, in order:
1. **`baseline_cnn`** -- the exact original architecture (Conv2D(16) → MaxPool → Flatten → Dense(256) → Dense(10), 16,780,490 params) so you have a fair, freshly-trained baseline to compare against.
2. **`smaller_cnn`** -- Technique 1. Three smaller conv blocks + `GlobalAveragePooling2D` instead of `Flatten`.
3. **`batchnorm_cnn`** -- Technique 2. Same smaller backbone, `BatchNormalization` after every Conv2D and Dense.
4. **`dropout_cnn`** -- Technique 3. Same smaller backbone, `Dropout(0.4)` after the Dense layer.
5. **`final_best_cnn`** -- Technique 4 (combined). Smaller backbone + BatchNorm + Dropout, trained with `ReduceLROnPlateau` + `EarlyStopping`.

**Why this design isolates each technique:** techniques 2 and 3 are each applied to the *same* smaller backbone (not to the giant baseline), so the accuracy difference you see between `smaller_cnn`, `batchnorm_cnn`, and `dropout_cnn` is attributable to BatchNorm or Dropout specifically, not to the confounded effect of also shrinking the model. The final model then stacks everything.

**On the epoch budget:** `MAX_EPOCHS=15` is just a ceiling. Every run uses `EarlyStopping(patience=4-5, restore_best_weights=True)`, so a model that plateaus at epoch 6 stops at epoch 6 with its best-epoch weights restored -- you get the same conclusions as training for 50 epochs, in a fraction of the time. `final_best_cnn` additionally uses `ReduceLROnPlateau` so it can keep inching down in loss with a smaller learning rate before EarlyStopping finally kicks in.

Outputs land in `outputs_lab4/`:
- `<model_name>_curves.png` -- train/val accuracy and loss curves per model
- `comparison_summary.png` -- bar charts of test accuracy, overfitting gap, and parameter count across all 5 models
- `results_summary.json` -- the same numbers in machine-readable form
- `best_model.keras` + `class_names.json` -- what Lab 5 loads

---

## 3. Lab 4 analysis (fill in your actual numbers from `results_summary.json`)

The script prints a table like this at the end -- paste your real numbers into your submitted report; the structure and reasoning below hold regardless of the exact figures:

```
Model              TestAcc  TrainAcc     Gap        Params  Epochs
baseline_cnn        59.xx%    99.xx%   40.xx%    16,780,490       9
smaller_cnn          xx.xx%    xx.xx%    x.xx%        ~45,000      xx
batchnorm_cnn        xx.xx%    xx.xx%    x.xx%        ~45,400      xx
dropout_cnn          xx.xx%    xx.xx%    x.xx%        ~45,000      xx
final_best_cnn       xx.xx%    xx.xx%    x.xx%        ~45,400      xx
```

**1. Was overfitting reduced?**
Yes. The overfitting gap (train accuracy minus validation accuracy) should shrink dramatically from the baseline's ~40 points down to single digits for `final_best_cnn`. This is the direct effect of removing the model's capacity to memorize the training set (fewer parameters) and adding explicit regularization (Dropout randomly disabling units, BatchNorm reducing internal covariate shift and adding a mild noise/regularizing effect).

**2. Which technique contributed most?**
In practice, going from `baseline_cnn` to `smaller_cnn` is usually the single biggest jump, because the original model's 16.78M parameters were almost entirely one oversized `Dense(256)` layer sitting on top of a flattened 65,536-length vector -- a layer with roughly 370x more parameters than the entire rest of the model combined. Cutting that (via smaller conv filters + GlobalAveragePooling2D) removes the model's ability to simply memorize pixel patterns per training image. BatchNorm and Dropout each give a further, smaller improvement on top, and the combined `final_best_cnn` should generalize the best of all five, with the LR schedule squeezing out the last bit of validation performance without overfitting further. (Confirm this ordering against your own run -- with augmentation absent and a modestly sized dataset, BatchNorm vs. Dropout's relative contribution can flip depending on batch size and dataset size.)

**3. How did the number of trainable parameters change?**
From 16,780,490 (baseline) down to roughly 45,000 (smaller/dropout variants) and ~45,400 (batchnorm/final, since BN adds a small number of scale+shift parameters per normalized layer). That's a >99.7% reduction, almost entirely from replacing `Flatten → Dense(256)` with `GlobalAveragePooling2D → Dense(64)`.

**4. How did the training/validation curves change?**
The baseline's curves show the classic overfitting signature: training accuracy climbs to ~99% while validation accuracy plateaus around 55-60% and validation loss starts *increasing* after a few epochs even as training loss keeps falling. The improved models should show training and validation accuracy tracking much closer together throughout training, and validation loss decreasing (or flattening) rather than turning upward -- i.e., no divergence between the two curves.

**5. Why does the final model perform better?**
Three complementary mechanisms are stacked: (a) a smaller hypothesis space (far fewer parameters) makes it mathematically harder for the network to memorize training-set-specific noise instead of learning generalizable features; (b) BatchNorm stabilizes and smooths the loss landscape, letting the optimizer take more consistent steps and acting as a mild regularizer; (c) Dropout forces the dense layer to not rely on any single unit, which approximates ensembling many sub-networks; and (d) the LR schedule lets the optimizer take large steps early and fine-tune with small steps later, while EarlyStopping prevents training past the point of diminishing (or negative) validation returns. Together these address both *causes* of the original overfitting: excess capacity and an optimization process with no signal to stop at the right time.

---

## 4. Running Lab 5

### Streamlit (recommended -- fastest to stand up)
```bash
streamlit run lab5_streamlit_app.py
```
Opens a browser tab where you upload an image and see the predicted class, confidence, and a top-3 bar chart.

### Flask (alternative)
```bash
python lab5_flask_app.py
```
Open `http://127.0.0.1:5000`. Same functionality, plain HTML form + inline template (no `templates/` folder needed for a lab submission).

Both apps:
1. Load `outputs_lab4/best_model.keras` and `outputs_lab4/class_names.json` produced by Lab 4 (run Lab 4 first).
2. Resize the uploaded image to 128x128 and rescale to `[0,1]`, matching training preprocessing exactly -- this consistency matters, a mismatched preprocessing pipeline is one of the most common reasons a "deployed" model performs worse than its reported test accuracy.
3. Run `model.predict`, sort the 10-class probability vector, and show the top-3.

---

## 5. Lab 5 report outline (what to write up)

1. **System overview** -- one diagram/paragraph: browser → Streamlit/Flask server → loaded Keras model → prediction → JSON/HTML response.
2. **Model used** -- name the winning architecture from Lab 4 (`final_best_cnn`), its test accuracy, and why it was chosen over the other 4.
3. **Preprocessing pipeline** -- resize to 128x128, RGB, `/255` normalization; note that this must exactly mirror training-time preprocessing.
4. **Interface walkthrough** -- screenshot of the upload form, screenshot of a prediction result with confidence + top-3.
5. **Framework choice justification** -- Streamlit: fastest for a data-science-style demo, built-in widgets, no HTML needed. Flask: more control over the HTTP layer / JSON API if the model needs to be consumed by another program rather than a human.
6. **Limitations & future work** -- no image augmentation robustness testing, single-image inference only (no batch endpoint), no authentication/rate limiting if this were to go to real production, and no model versioning if you retrain the CNN later.

---

## 6. If you're short on time before a deadline

- Reduce `IMG_SIZE` to 64 (both in Lab 4's config and Lab 5's `IMG_SIZE`) to roughly 4x the training speed at some accuracy cost -- fine for demonstrating the *relative* comparison the assignment asks for.
- Lower `MAX_EPOCHS` further (e.g. 8) -- EarlyStopping will still find the right stopping point for most of these models on a typical clothing dataset.
- Run on Colab with a free GPU runtime (Runtime → Change runtime type → GPU) rather than a local CPU.
