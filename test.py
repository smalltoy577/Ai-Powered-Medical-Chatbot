import re
import random
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from difflib import get_close_matches
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def show_model_performance():
    """Display model performance metrics and visualizations in terminal"""
    
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE ANALYSIS")
    print("="*70)
    
    # 1. Accuracy on training and test sets
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\n✅ Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"✅ Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    # 2. Classification Report
    print("\n" + "-"*70)
    print("📋 DETAILED CLASSIFICATION REPORT (Test Set)")
    print("-"*70)
    print(classification_report(y_test, test_pred, target_names=le.classes_, zero_division=0))
    
    # 3. Confusion Matrix (show first 10x10 for readability)
    print("\n" + "-"*70)
    print("🎯 CONFUSION MATRIX PREVIEW (First 10 diseases)")
    print("-"*70)
    cm = confusion_matrix(y_test, test_pred)
    
    # Show only first 10x10
    cm_preview = cm[:10, :10]
    disease_names = le.classes_[:10]
    
    # Print header
    print("\n" + " "*15, end="")
    for name in disease_names:
        print(f"{name[:8]:>10s}", end="")
    print("\n" + "-"*140)
    
    # Print matrix
    for i, name in enumerate(disease_names):
        print(f"{name[:12]:>15s}", end="")
        for j in range(len(disease_names)):
            print(f"{cm_preview[i][j]:>10d}", end="")
        print()
    
    # 4. Model Statistics
    print("\n" + "-"*70)
    print("📈 MODEL STATISTICS")
    print("-"*70)
    print(f"Total Training Samples: {len(x_train)}")
    print(f"Total Testing Samples: {len(x_test)}")
    print(f"Number of Features (Symptoms): {len(cols)}")
    print(f"Number of Disease Classes: {len(le.classes_)}")
    print(f"Number of Trees in Random Forest: {model.n_estimators}")
    
    # 5. Per-class accuracy
    print("\n" + "-"*70)
    print("🎯 PER-DISEASE ACCURACY (Top 10 Most Common)")
    print("-"*70)
    
    disease_counts = Counter(le.inverse_transform(y_test))
    top_diseases = disease_counts.most_common(10)
    
    for disease_name, count in top_diseases:
        # Get predictions for this disease
        disease_idx = le.transform([disease_name])[0]
        mask = y_test == disease_idx
        disease_pred = test_pred[mask]
        disease_true = y_test[mask]
        
        if len(disease_true) > 0:
            acc = accuracy_score(disease_true, disease_pred)
            print(f"{disease_name:30s} → Accuracy: {acc*100:6.2f}% (Samples: {count})")
    
    print("\n" + "="*70)
    
    # 6. Generate and save visualizations
    print("\n📊 Generating visualization plots...")
    
    # Plot 1: Confusion Matrix Heatmap (first 15 diseases for visibility)
    plt.figure(figsize=(14, 12))
    cm_subset = cm[:15, :15]
    disease_subset = le.classes_[:15]
    
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                xticklabels=disease_subset, yticklabels=disease_subset,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Disease')
    plt.ylabel('Actual Disease')
    plt.title('Confusion Matrix (First 15 Diseases)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: confusion_matrix.png")
    
    
    # Plot 3: Disease Distribution
    plt.figure(figsize=(12, 6))
    disease_counts_all = Counter(le.inverse_transform(y_test))
    top_15 = disease_counts_all.most_common(15)
    diseases, counts = zip(*top_15)
    
    plt.bar(range(len(diseases)), counts, color='coral', alpha=0.8)
    plt.xticks(range(len(diseases)), diseases, rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.title('Disease Distribution in Test Set (Top 15)')
    plt.tight_layout()
    plt.savefig('disease_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: disease_distribution.png")
    
    print("\n✅ All visualizations saved successfully!")
    print("="*70 + "\n")


def initialize():
    """Load data, train model, evaluate, and run reachability check.
    This is done inside a function so importing this module doesn't run heavy side-effects
    or print duplicate output when the module is imported prior to being executed.
    """
    global training, testing, cols, x, y, le, x_train, x_test, y_train, y_test
    global model, training_vectors, training_labels

    # ------------------ Load Data ------------------
    training = pd.read_csv("Data/Training.csv")
    testing = pd.read_csv("Data/Testing.csv")

    # Clean duplicate column names and normalize column formatting
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
    training.columns = training.columns.str.strip().str.replace(' ', '_')
    testing.columns = testing.columns.str.strip().str.replace(' ', '_')
    training = training.loc[:, ~training.columns.duplicated()]
    testing = testing.loc[:, ~testing.columns.duplicated()]

    # Normalize label strings (strip whitespace) to avoid mismatches with Master Data
    if 'prognosis' in training.columns:
        training['prognosis'] = training['prognosis'].astype(str).str.strip()
    if 'prognosis' in testing.columns:
        testing['prognosis'] = testing['prognosis'].astype(str).str.strip()

    # Features and labels
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    # initialize symptom -> index mapping
    global symptoms_dict
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

    # Encode target
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # ------------------ Model Training ------------------
    print("\n🚀 Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    print("✅ Model training completed successfully!")

    # Prepare arrays for a simple nearest-neighbor fallback (binary vectors)
    try:
        training_vectors = training[cols].fillna(0).astype(int).values
        training_labels = training['prognosis'].astype(str).tolist()
    except Exception:
        training_vectors = np.array([])
        training_labels = []

    # ------------------ Model Evaluation ------------------
    print("\n🧠 Testing the model on unseen data...")
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model testing completed! Accuracy: {acc * 100:.2f}%")

    # ------------------ Validation: Reachability Check ------------------
    def validate_reachability():
        """Check whether each disease label is reachable by at least one training symptom vector.
        This helps detect label mismatch or columns problems causing some diseases never to be predicted.
        """
        unreachable = []
        if training_vectors.size == 0:
            print("⚠️ No training vectors available for reachability check.")
            return

        for label in le.classes_:
            # get all training rows with this label
            rows = training[training['prognosis'] == label]
            if rows.empty:
                unreachable.append(label)
                continue

            found = False
            for _, r in rows.iterrows():
                vec = r[:-1].values.astype(int)
                # if model assigns a non-zero probability to this label for this exact vector
                try:
                    # construct a DataFrame so feature names match what the model was trained on
                    try:
                        df_vec = pd.DataFrame([vec], columns=cols)
                        proba = model.predict_proba(df_vec)[0]
                    except Exception:
                        # fallback to older array-based call if DataFrame construction fails
                        proba = model.predict_proba([vec])[0]
                    pred = le.inverse_transform([int(np.argmax(proba))])[0]
                    if pred == label or proba.max() > 0.05:
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                unreachable.append(label)

        if unreachable:
            print("\n⚠️ Reachability warning: the following labels may be unreachable or under-represented:")
            for l in unreachable:
                print(" - ", l)
        else:
            print("\n✅ All labels appear reachable by at least one training example (quick check).")

    validate_reachability()
    
    # Show performance metrics and graphs
    show_model_performance()


# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
# will be initialized inside initialize() after data is loaded
symptoms_dict = {}

def getDescription():
    with open("Master Data/symptom_Description.csv") as csv_file:
        for row in csv.reader(csv_file):
            if not row:
                continue
            key = row[0].strip()
            description_list[key] = row[1] if len(row) > 1 else ''

def getSeverityDict():
    with open("Master Data/Symptom_severity.csv") as csv_file:
        for row in csv.reader(csv_file):
            if not row:
                continue
            try:
                severityDictionary[row[0].strip()] = int(row[1])
            except Exception:
                # ignore malformed lines
                pass

def getprecautionDict():
    with open("Master Data/symptom_precaution.csv") as csv_file:
        for row in csv.reader(csv_file):
            if not row:
                continue
            key = row[0].strip()
            # Some rows may have fewer than 4 suggestions
            vals = [c for c in row[1:5] if c is not None]
            precautionDictionary[key] = vals

# ------------------ Symptom Extractor ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    # dataset uses 'diarrhoea'
    "loose motion": "diarrhoea",
    "motions": "diarrhoea",
    # dataset has 'high_fever' and 'mild_fever' columns; map generic 'fever' to 'high_fever'
    "high temperature": "high_fever",
    "temperature": "high_fever",
    "feaver": "high_fever",
    "coughing": "cough",
    # map throat pain to a likely column
    "throat pain": "throat_irritation",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    """Robust symptom extractor:
    - normalizes input
    - applies synonym mapping
    - exact phrase matching
    - token-overlap matching (handles reordered words)
    - fuzzy phrase matching for typos
    Returns list of symptom column names (matching the training columns)
    """
    extracted = set()
    text = user_input.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = re.findall(r"\w+", text)

    # 1. Synonym replacement (phrase-based)
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.add(mapped)

    # Prepare phrase list for all_symptoms
    symptom_phrases = {sym: sym.replace("_", " ") for sym in all_symptoms}

    # 2. Exact phrase match and token-overlap
    for sym, phrase in symptom_phrases.items():
        if phrase in text:
            extracted.add(sym)
            continue

        # token overlap: require at least half of phrase tokens to match
        phrase_tokens = re.findall(r"\w+", phrase)
        if not phrase_tokens:
            continue
        common = set(phrase_tokens) & set(tokens)
        if len(common) >= max(1, (len(phrase_tokens) // 2)):
            extracted.add(sym)
            continue

        # 3. fuzzy phrase match (for typos or near matches)
        close = get_close_matches(phrase, [text], n=1, cutoff=0.7)
        # Note: get_close_matches expects a sequence of choices -- we compare phrase to the full text
        # but since text is long, also try matching phrase against joined tokens ngrams
        if not close:
            # build ngrams from tokens up to length of phrase tokens
            n = len(phrase_tokens)
            joined = " ".join(tokens)
            # fallback: compare phrase to joined text with lower threshold
            close2 = get_close_matches(phrase, [joined], n=1, cutoff=0.7)
            if close2:
                extracted.add(sym)

    # 4. Word-level fuzzy matching (single word typos)
    for word in tokens:
        close = get_close_matches(word, [p for p in symptom_phrases.values()], n=1, cutoff=0.8)
        if close:
            # find the symptom corresponding to this phrase
            for sym, phrase in symptom_phrases.items():
                if phrase == close[0]:
                    extracted.add(sym)
                    break

    return list(extracted)

# ------------------ Prediction ------------------
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    # If user provided no recognized symptoms, return unknown
    if input_vector.sum() == 0:
        return "Unknown", 0.0, np.zeros(len(le.classes_))

    # ensure we pass a DataFrame with matching column names to avoid feature-name warnings
    try:
        df_input = pd.DataFrame([input_vector], columns=cols)
        pred_proba = model.predict_proba(df_input)[0]
    except Exception:
        pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    top_prob = pred_proba[pred_class]
    disease = le.inverse_transform([pred_class])[0]

    # Fallback: if model is not confident, pick the most similar training example (nearest neighbor)
    if top_prob < 0.30:
        try:
            sims = training_vectors.dot(input_vector)
            best_idx = int(np.argmax(sims))
            nn_label = training_labels[best_idx]
            # similarity normalized by number of input symptoms
            sim_score = sims[best_idx] / max(1, input_vector.sum())
            # choose the label from NN if it's a better match
            if sim_score > top_prob:
                disease = nn_label
                confidence = round(sim_score * 100, 2)
            else:
                confidence = round(top_prob * 100, 2)
        except Exception:
            confidence = round(top_prob * 100, 2)
    else:
        confidence = round(top_prob * 100, 2)

    return disease, confidence, pred_proba

# ------------------ Empathy Quotes ------------------
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Chatbot ------------------
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print("🤖 Welcome to HealthCare ChatBot")
    print("Hello! Please answer a few questions so I can understand your condition better.")

    # Basic info
    name = input("👉 What is your name? : ")
    age = input("👉 Please enter your age: ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    # Initial symptoms
    symptoms_input = input("👉 Describe your symptoms in a sentence (e.g., 'I have fever and stomach pain'): ")
    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("❌ Sorry, I could not detect valid symptoms. Please try again with more details.")
        return

    print(f"✅ Detected symptoms: {', '.join(symptoms_list)}")

    num_days = int(input("👉 For how many days have you had these symptoms? : "))
    severity_scale = int(input("👉 On a scale of 1–10, how severe do you feel your condition is? : "))
    pre_exist = input("👉 Do you have any pre-existing conditions (e.g., diabetes, hypertension)? : ")
    lifestyle = input("👉 Do you smoke, drink alcohol, or have irregular sleep? : ")
    family = input("👉 Any family history of similar illness? : ")

    # ---------------- Initial Prediction ----------------
    disease, confidence, proba = predict_disease(symptoms_list)

    # ---------------- Guided Questions (Disease-specific) ----------------
    print("\n🤔 Let me ask you some more questions related to", disease)
    # If the predicted disease is 'Unknown' or there are no training rows for it,
    # skip the disease-specific guided questions to avoid indexing errors.
    d_rows = training[training['prognosis'] == disease]
    if disease == "Unknown" or d_rows.empty:
        print("I don't have a specific disease match to ask targeted questions. I'll ask no additional disease-specific questions.")
        disease_symptoms = []
    else:
        disease_symptoms = list(d_rows.iloc[0][:-1].index[d_rows.iloc[0][:-1] == 1])

    asked = 0
    for sym in disease_symptoms:
        if sym not in symptoms_list and asked < 8:
            ans = input(f"👉 Do you also have {sym.replace('_',' ')}? (yes/no): ").strip().lower()
            if ans == "yes":
                symptoms_list.append(sym)
            asked += 1

    # ---------------- Final Prediction ----------------
    disease, confidence, proba = predict_disease(symptoms_list)

    print("\n---------------- Result ----------------")
    print(f"🩺 Based on your answers, you may have **{disease}**")
    print(f"🔎 Confidence: {confidence}%")
    print(f"📖 About: {description_list.get(disease, 'No description available.')}")

    if disease in precautionDictionary:
        print("\n🛡️ Suggested precautions:")
        for i, prec in enumerate(precautionDictionary[disease], 1):
            print(f"{i}. {prec}")

    # Random empathy quote
    print("\n💡 " + random.choice(quotes))
    print("\nThank you for using the chatbot. Wishing you good health,", name + "!")

# ------------------ Run ------------------
if __name__ == "__main__":
    # initialize model and data, then start chatbot
    initialize()
    chatbot()