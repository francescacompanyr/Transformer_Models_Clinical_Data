# TFG_submit

# HealthTab-Transformer: Models Transformer per a Dades Tabulars de Salut

Aquest projecte explora l'aplicació d'arquitectures basades en Transformers per a tasques de classificació sobre dades de salut tabulars. La idea principal és tractar cada característica (feature) d'un pacient com un "token", permetent al model capturar interaccions complexes entre elles.

El repositori inclou experiments amb dos conjunts de dades:
1.  **RadioLung**: Un dataset privat (format Excel) per a la predicció de nòduls pulmonars (Benigne/Malígn).
2.  **CDC Diabetes**: Un dataset públic de [Hugging Face](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators) per a la predicció de la diabetis.

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
*   **Models de Base**: Inclou un notebook amb models clàssics (Random Forest, XGBoost) per comparar resultats.

## Estructura del Projecte

```
.
├── data/                  # Directori per als fitxers de dades (ignorat per git)
├── dataset/               # Càrrega, preprocessat i partició de dades
│   ├── radiolung.py
│   └── splitter.py
├── framework/             # Components centrals del framework
│   ├── tokenizer.py
│   └── trainer.py
├── models/                # Arquitectures de models
│   ├── base_model.py
│   └── transformer.py
├── notebooks/             # Notebooks per anàlisi i models clàssics
│   └── random-forest_xgboost.ipynb
├── training/              # Scripts principals per executar experiments
│   ├── kfold.py
│   ├── kfold_radiolung2.py
│   └── finetuning.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Instal·lació

1.  **Clonar el repositori:**
    ```bash
    git clone https://github.com/francescacompanyr/TFG_submit.git
    cd TFG_submit
    ```

2.  **Crear un entorn virtual (recomanat):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # A Windows: venv\Scripts\activate
    ```

3.  **Instal·lar les dependències:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar les dades (per al dataset RadioLung):**
    Col·loca els quatre fitxers `.xlsx` del dataset RadioLung dins del directori `data/`.

5.  **Iniciar sessió a Weights & Biases:**
    Si vols fer seguiment dels experiments, necessitaràs un compte a `wandb`.
    ```bash
    wandb login
    ```

## Ús

Els experiments principals es poden llançar executant els scripts del directori `training/`. La configuració de cada experiment es troba dins del bloc `if __name__ == "__main__":` de cada fitxer.

### Configuració d'un Experiment

Dins de cada script d'entrenament, trobaràs un diccionari `info_runing` que controla el comportament del model:

*   `separat`: Controla l'arquitectura del Transformer.
    *   `0`: `Transformer` (totes les features juntes).
    *   `1`: `TransformerSeparated` (numèric vs. categòric).
    *   `2`: `TransformerGroup` (agrupació manual definida a `dic_group`).
*   `emb_dim`: La dimensió dels embeddings de les característiques.
*   `pretrained`: `True` si s'utilitza un model de llenguatge pre-entrenat (com ClinicalBERT o DistilBERT) per a les característiques textuals. `False` per entrenar els embeddings des de zero.
*   `corpus tokenizer`: Estratègia de tokenització.
    *   `0`: Un tokenizer per a cada característica.
    *   `1`: Un tokenizer comú per a totes les característiques categòriques i un altre per a les numèriques.
    *   `2`: Un tokenizer per a cada grup de característiques definit a `dic_group`.
*   `contextual`: `True` per activar la tokenització contextual (p. ex., `1` -> `"Fumador"`).

### Exemples d'Execució

*   **Entrenar i avaluar amb K-Fold en el dataset de Diabetis:**
    ```bash
    python training/kfold.py
    ```
    (Modifica el diccionari `info_runing` dins del fitxer per provar diferents arquitectures).

*   **Fer Fine-Tuning en el dataset RadioLung:**
    Aquest script primer entrena un model des de zero i després en fa un altre amb fine-tuning a partir d'un checkpoint pre-entrenat.
    ```bash
    python training/finetuning.py
    ```
    Assegura't que la ruta a `checkpoint_path` dins de `finetune_config` sigui correcta.

*   **Executar els models de base (Random Forest/XGBoost):**
    Obre i executa les cel·les del notebook `notebooks/random-forest_xgboost.ipynb` utilitzant Jupyter.

## Contribució

Les contribucions són benvingudes. Si us plau, obre un *issue* per discutir els canvis proposats o envia un *pull request*.

## Llicència

Aquest projecte està sota la llicència MIT. Consulta el fitxer `LICENSE` per a més detalls.
