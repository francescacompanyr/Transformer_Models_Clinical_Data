
# Anàlisi del comportament dels Transformers amb dades tabulars en l’àmbit clínic 

Aquest projecte explora l'aplicació d'arquitectures basades en Transformers per a tasques de classificació sobre dades tabulars de salut . La idea principal és tractar cada característica  d'un pacient com un "token", permetent al model capturar interaccions complexes entre elles.

El repositori inclou experiments amb dos conjunts de dades:
1.  **RadioLung**: Un dataset privat  per a la predicció de nòduls pulmonars.
2.  **DiabetesDisease**: Un dataset públic de [Hugging Face](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators) per a la predicció de la diabetis.

El projecte està implementat utilitzant PyTorch i s'integra amb `wandb` per al seguiment d'experiments.

## Característiques Principals

*   **Arquitectures Flexibles**:
    *   `Transformer`: Un model únic que processa totes les característiques juntes.
    *   `TransformerSeparated`: Processa característiques numèriques i categòriques en Transformers separats abans de combinar-les.
    *   `TransformerGroup`: Permet agrupar manualment les característiques (p. ex., "dades personals", "analítiques") i assignar un Transformer a cada grup.
*   **Validació Creuada (K-Fold)**: Scripts per entrenar i avaluar els models de manera robusta mitjançant validació creuada estratificada.
*   **Fine-Tuning**: Capacitat de pre-entrenar un model en un dataset gran (com el de la diabetis) i fer fine-tuning en un de més petit (RadioLung).
*   **Tokenització Contextual**: Opció per convertir variables numèriques o codificades (com `0` o `1`) en frases més descriptives (p. ex., "No fuma", "Pressió arterial alta") abans de la tokenització.
*   **Seguiment d'Experiments**: Integració total amb [Weights & Biases](https://wandb.ai) per visualitzar mètriques, pèrdues i configuracions.
*   **Models ML**: Inclou un notebook amb models clàssics (Random Forest, XGBoost) per comparar resultats.
*   **Jupiter Notebooks**: Inclou notebooks de diferentes proves que s'han realitzat durant el treball.

## Estructura del Projecte

```
.
├── data/                  # Directori per als fitxers de dades (ignorat per git)
├── dataset/               # Càrrega, preprocessat i partició de dades
│   ├── radiolung.py
│   └── splitter.py
├── models/                # Arquitectures de models
│   ├── base_model.py
│   └── transformer.py
├── notebooks/ # Notebooks per anàlisi i models clàssics
│   ├── ct-rate.ipynb
│   ├── improve_dataset_set.ipynb
│   ├── radioLung_prova.ipynb
│   ├── testing_pyhealth.ipynb
│   └── random-forest_xgboost.ipynb
├── .gitignore
├── tokenizer.py
├── trainer.py
├── kfold.py
├── kfold_radiolung2.py
├── finetuning.py
├── main_test.py
├── main_test_dataset_diabetes.py
├── tokenizer_visualitzacio.py
├── main_test_dataset_diabetes.py
├── main_test_dataset_diabetes.py
├── README.md
└── requirements.txt

```


### Configuració d'un Experiment

Dins de cada script d'entrenament, trobaràs un diccionari `info_runing` que controla el comportament del model:

*   `separat`: Controla l'arquitectura del Transformer.
    *   `0`: `Transformer` (totes les features juntes).
    *   `1`: `TransformerSeparated` (numèric vs. categòric).
    *   `2`: `TransformerGroup` (agrupació manual definida a `dic_group`).
*   `emb_dim`: La dimensió dels embeddings de les característiques.
*   `pretrained`: `True` si s'utilitza un model de llenguatge pre-entrenat per a les característiques textuals. `False` per entrenar els embeddings des de zero.
*   `corpus tokenizer`: Estratègia de tokenització.
    *   `0`: Un tokenizer per a cada característica.
    *   `1`: Un tokenizer comú per a totes les característiques categòriques i un altre per a les numèriques.
    *   `2`: Un tokenizer per a cada grup de característiques definit a `dic_group`.
*   `contextual`: `True` per activar la tokenització contextual (p. ex., `1` -> `"Fumador"`).

### Exemples d'Execució

*   **Entrenar i avaluar amb K-Fold en el dataset de Diabetis:**
    ```bash
    python kfold.py
    ```
    (Modifica el diccionari `info_runing` dins del fitxer per provar diferents arquitectures).


*   **Executar els models de base (Random Forest/XGBoost):**
    Obre i executa les cel·les del notebook `notebooks/random-forest_xgboost.ipynb` utilitzant Jupyter. Caldrà algunes modificacions dependent del dataset que s'utilitzi.


