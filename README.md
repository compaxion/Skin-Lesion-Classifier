# Skin Lesion Classification Dashboard — with Authentication

A Streamlit dashboard for skin-lesion classification that uses a trained Keras multi-modal model (EfficientNetB2 + patient metadata) to predict the lesion type from a dermoscopic image. This version adds a SQLite-backed authentication layer and per-user upload history.

## Features

- **User registration & login** with bcrypt-hashed passwords
- **Session management** via `st.session_state`
- **Per-user prediction history** — every upload is persisted, scoped to the owning user
- **Detail view + delete** for individual history records
- **Logout button** in the sidebar that wipes the entire session

## Available Classes

The model predicts the following 7 classes:

* **0:** Actinic keratosis *(Cancer - Pre)*
* **1:** Basal cell carcinoma *(Cancer)*
* **2:** Benign keratosis *(Benign)*
* **3:** Dermatofibroma *(Benign)*
* **4:** Melanoma *(Cancer)*
* **5:** Melanocytic nevi *(Benign - Mole)*
* **6:** Vascular lesion *(Benign)*

## Project Layout

```
.
├── app.py                     # Streamlit entry — auth gate + router only
├── auth.py                    # register / login / password hashing (bcrypt)
├── db.py                      # SQLite connection + schema + scoped queries
├── skin_lesion_dashboard.py   # THE dashboard — 6 tabs incl. My History
├── history.py                 # history-tab renderer (called from the dashboard)
├── requirements.txt
├── best_skin_model.keras          # EfficientNetB2 multi-modal model
└── data/                      # created on first run (gitignored)
    ├── app.db                 # SQLite database
    └── uploads/{user_id}/{uuid}.{ext}
```

There is now a **single** dashboard. The old standalone classifier UI that
used to live in `app.py` has been folded into `skin_lesion_dashboard.py`,
which renders behind the login gate. `app.py` no longer contains any
classifier UI — it only handles authentication and then calls
`render_dashboard(user)`.

### Dashboard tabs
1. **Analyze** — upload an image, run inference; real (non-demo) results are saved to history.
2. **Metrics & Trade-offs**, **Model Docs**, **Class Guide**, **About** — reference material (unchanged).
3. **My History** — the signed-in user's past uploads, with a detail view and per-record delete.

## Installation & Usage

1. **Clone the Repository:**
This project uses Git LFS for large files. Make sure you have Git LFS installed on your system before cloning.

    ```bash
    git clone https://github.com/compaxion/Skin-Lesion-Classifier.git
    cd <project_folder>
    git lfs install
    git lfs pull
    ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
4. On first launch, click the **Register** tab, create an account, and you'll be logged in automatically.

## Database Schema

```sql
users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_path         TEXT    NOT NULL,
    original_filename  TEXT    NOT NULL,
    predicted_class    TEXT    NOT NULL,
    confidence         REAL    NOT NULL,
    probabilities_json TEXT    NOT NULL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Security Notes

- Passwords are hashed with **bcrypt** (cost factor 12) — never stored in plaintext.
- All SQL uses parameterized queries to prevent injection.
- A constant-time-style dummy bcrypt comparison is run on unknown-username login attempts to prevent username enumeration via response timing.
- Every read/write on `predictions` is scoped to the requesting user's `id` to prevent IDOR (one user reading another user's data).
- Uploaded files are stored under UUID filenames, not the user-supplied name, to neutralise path-traversal attacks.
- On logout, **all** session state is wiped — not just the auth flag — so leftover widget state cannot leak between accounts on a shared browser.

## Notes

- The dashboard's **Demo mode** (sidebar) produces simulated output when the `.keras` model is absent. Demo results are intentionally **not** saved to history — only real model inferences are persisted.
- **Inference Pipeline:** The model uses a multi-branch architecture:
  - **Image:** Resized to 224×224 RGB, converted to float32 (no `/255.0` normalization).
  - **Metadata:** Age (float), Sex (string), and Localization (string).
