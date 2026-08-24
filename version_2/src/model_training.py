import os
import pickle

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = "models"

X_TRAIN_PATH = os.path.join(
    MODELS_DIR,
    "X_train.pkl"
)

Y_TRAIN_PATH = os.path.join(
    MODELS_DIR,
    "y_train.pkl"
)

CHECKPOINT_PATH = os.path.join(
    MODELS_DIR,
    "checkpoint.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print(
        "\nLoading training data..."
    )

    if not os.path.exists(X_TRAIN_PATH):

        raise FileNotFoundError(
            "Training feature file not found:\n"
            + X_TRAIN_PATH
        )

    if not os.path.exists(Y_TRAIN_PATH):

        raise FileNotFoundError(
            "Training target file not found:\n"
            + Y_TRAIN_PATH
        )

    with open(
        X_TRAIN_PATH,
        "rb"
    ) as file:

        X_train = pickle.load(
            file
        )

    with open(
        Y_TRAIN_PATH,
        "rb"
    ) as file:

        y_train = pickle.load(
            file
        )

    print(
        "X_train shape:",
        X_train.shape
    )

    print(
        "y_train length:",
        len(y_train)
    )

    if X_train.shape[0] != len(y_train):

        raise ValueError(
            "X_train and y_train have different numbers "
            "of samples."
        )

    return (
        X_train,
        y_train
    )


# ============================================================
# MODEL FEATURE COMPATIBILITY
# ============================================================

def get_model_feature_count(
    model
):

    if hasattr(
        model,
        "n_features_in_"
    ):

        return model.n_features_in_

    return None


def is_model_compatible(
    model,
    expected_features
):

    model_features = get_model_feature_count(
        model
    )

    if model_features is None:

        print(
            "Model does not expose n_features_in_."
        )

        return False

    print(
        "Model features:",
        model_features
    )

    print(
        "Current features:",
        expected_features
    )

    return (
        model_features
        == expected_features
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        print(
            "\nNo checkpoint found."
        )

        print(
            "Training models from scratch."
        )

        return {}

    print(
        "\nLoading checkpoint..."
    )

    try:

        with open(
            CHECKPOINT_PATH,
            "rb"
        ) as file:

            models = pickle.load(
                file
            )

    except Exception as error:

        print(
            "\nWARNING: Could not load checkpoint."
        )

        print(
            "Reason:",
            error
        )

        print(
            "Starting with an empty checkpoint."
        )

        return {}

    if not isinstance(
        models,
        dict
    ):

        print(
            "\nWARNING: Checkpoint does not contain "
            "a model dictionary."
        )

        print(
            "Starting with an empty checkpoint."
        )

        return {}

    print(
        "Checkpoint loaded."
    )

    print(
        "Models found:",
        list(models.keys())
    )

    return models


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    models
):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    temporary_path = (
        CHECKPOINT_PATH
        + ".tmp"
    )

    with open(
        temporary_path,
        "wb"
    ) as file:

        pickle.dump(
            models,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    os.replace(
        temporary_path,
        CHECKPOINT_PATH
    )

    print(
        "Checkpoint saved."
    )


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    y_train
):

    expected_features = X_train.shape[1]

    print(
        "\nCurrent feature count:",
        expected_features
    )

    models = load_checkpoint()

    # --------------------------------------------------------
    # MODEL DEFINITIONS
    # --------------------------------------------------------

    model_list = [

        (
            "logistic_regression",

            LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "linear_svm",

            LinearSVC(
                class_weight="balanced"
            )
        )
    ]

    # --------------------------------------------------------
    # PROCESS MODELS
    # --------------------------------------------------------

    for name, new_model in tqdm(
        model_list,
        desc="Training Models"
    ):

        # ----------------------------------------------------
        # CHECK EXISTING MODEL
        # ----------------------------------------------------

        if name in models:

            existing_model = models[name]

            print(
                "\n"
                + "-" * 60
            )

            print(
                "Existing model found:",
                name
            )

            # ------------------------------------------------
            # FEATURE COMPATIBILITY CHECK
            # ------------------------------------------------

            compatible = is_model_compatible(
                existing_model,
                expected_features
            )

            if compatible:

                print(
                    "✅ Existing model is compatible."
                )

                print(
                    "Skipping retraining:",
                    name
                )

                continue

            # ------------------------------------------------
            # INCOMPATIBLE MODEL
            # ------------------------------------------------

            print(
                "⚠️ Existing model is incompatible "
                "with current features."
            )

            print(
                "Retraining:",
                name
            )

            # Replace stale model
            models.pop(
                name,
                None
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print(
            "\n🚀 Training:",
            name
        )

        model = new_model

        model.fit(
            X_train,
            y_train
        )

        print(
            "✅ Finished:",
            name
        )

        # ----------------------------------------------------
        # VERIFY TRAINED MODEL
        # ----------------------------------------------------

        trained_features = get_model_feature_count(
            model
        )

        if trained_features != expected_features:

            raise RuntimeError(
                "Trained model feature count does not "
                "match training data.\n"
                "Model: "
                + name
                + "\n"
                "Expected: "
                + str(expected_features)
                + "\n"
                "Actual: "
                + str(trained_features)
            )

        print(
            "Verified:",
            name,
            "expects",
            trained_features,
            "features."
        )

        # ----------------------------------------------------
        # UPDATE CHECKPOINT
        # ----------------------------------------------------

        models[name] = model

        save_checkpoint(
            models
        )

        print(
            "Checkpoint updated after:",
            name
        )

    return models


# ============================================================
# SAVE FINAL MODELS
# ============================================================

def save_models(
    models,
    expected_features
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SAVING FINAL MODELS"
    )

    print(
        "=" * 60
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    required_models = [
        "logistic_regression",
        "linear_svm"
    ]

    for name in required_models:

        if name not in models:

            raise RuntimeError(
                "Required model is missing:\n"
                + name
            )

        model = models[name]

        # ----------------------------------------------------
        # FINAL COMPATIBILITY CHECK
        # ----------------------------------------------------

        model_features = get_model_feature_count(
            model
        )

        if model_features != expected_features:

            raise RuntimeError(
                "Cannot save incompatible model:\n"
                + name
                + "\nExpected features: "
                + str(expected_features)
                + "\nModel features: "
                + str(model_features)
            )

        path = os.path.join(
            MODELS_DIR,
            name + ".pkl"
        )

        temporary_path = (
            path
            + ".tmp"
        )

        # ----------------------------------------------------
        # ATOMIC SAVE
        # ----------------------------------------------------

        with open(
            temporary_path,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(
            temporary_path,
            path
        )

        print(
            "Saved:",
            path
        )

        # ----------------------------------------------------
        # VERIFY FILE
        # ----------------------------------------------------

        if not os.path.exists(
            path
        ):

            raise RuntimeError(
                "Model file was not created:\n"
                + path
            )

        if os.path.getsize(
            path
        ) == 0:

            raise RuntimeError(
                "Model file is empty:\n"
                + path
            )

    print(
        "\n✅ All final models saved successfully."
    )


# ============================================================
# VERIFY SAVED MODELS
# ============================================================

def verify_saved_models(
    expected_features
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VERIFYING SAVED MODELS"
    )

    print(
        "=" * 60
    )

    model_names = [
        "logistic_regression",
        "linear_svm"
    ]

    for name in model_names:

        path = os.path.join(
            MODELS_DIR,
            name + ".pkl"
        )

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                "Saved model not found:\n"
                + path
            )

        with open(
            path,
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        model_features = get_model_feature_count(
            model
        )

        print(
            "\nModel:",
            name
        )

        print(
            "Expected features:",
            expected_features
        )

        print(
            "Model features:",
            model_features
        )

        if model_features != expected_features:

            raise RuntimeError(
                "Saved model has incompatible "
                "feature dimensions:\n"
                + name
            )

        print(
            "✅ Model verification passed."
        )

    print(
        "\nAll saved models are compatible "
        "with current features."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    X_train, y_train = load_data()

    expected_features = X_train.shape[1]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING MODELS"
    )

    print(
        "=" * 60
    )

    models = train_models(
        X_train,
        y_train
    )

    save_models(
        models,
        expected_features
    )

    verify_saved_models(
        expected_features
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        "\nAll models are trained against:"
    )

    print(
        expected_features,
        "features"
    )

    print(
        "\nNext stage: evaluation.py"
    )


if __name__ == "__main__":

    main()