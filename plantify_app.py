import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
import csv
import webbrowser
import os
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Streamlit app
st.set_page_config(page_title="South Indian Medicinal Herb Classifier", page_icon="🌿", layout="wide")
st.markdown("""
<style>
/* Fade-in animation for welcome page */
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

/* Slide-in animation for classification inputs */
@keyframes slideIn {
    0% { transform: translateX(-100px); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}

.fade-in {
    animation: fadeIn 1.5s ease-in-out;
}

.slide-in {
    animation: slideIn 0.8s ease-in-out;
}

/* Button hover effect */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}

.stSelectbox select { font-size: 16px; }
.stTextInput > div > input { font-size: 16px; }
.reportview-container { background: linear-gradient(to bottom, #e6f3e6, #ffffff); }
.sidebar .sidebar-content { background-color: #f0f8f0; }

/* Center logo and name in footer */
.footer-container {
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# Function to load and encode the logo as base64
def get_base64_image(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Logo file not found at {file_path}. Using placeholder instead.")
        return "https://via.placeholder.com/30"
    except Exception as e:
        st.error(f"Error loading logo: {str(e)}. Using placeholder instead.")
        return "https://via.placeholder.com/30"

# Define logo path and encode it at the top to ensure it's available
logo_path = "MY_LOGO.png"
logo_base64 = get_base64_image(logo_path)

# Expanded dataset with 100 South Indian medicinal herbs
characteristics_to_species = {
    ("opposite", "actinomorphic", "5", "superior", "herb", "nutlet", "simple", "spike"): ("Lamiaceae", "Ocimum tenuiflorum"),
    ("opposite", "zygomorphic", "5", "superior", "shrub", "capsule", "simple", "raceme"): ("Acanthaceae", "Justicia adhatoda"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "schizocarp", "palmate", "umbel"): ("Apiaceae", "Centella asiatica"),
    ("alternate", "actinomorphic", "4", "superior", "herb", "capsule", "simple", "cyme"): ("Nyctaginaceae", "Boerhavia diffusa"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "lobed", "head"): ("Asteraceae", "Eclipta prostrata"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Alternanthera sessilis"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "capsule", "palmate", "cyme"): ("Malvaceae", "Abutilon indicum"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Bacopa monnieri"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "simple", "spike"): ("Lamiaceae", "Leucas aspera"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "linear", "panicle"): ("Acanthaceae", "Andrographis paniculata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Spilanthes acmella"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Eryngium foetidum"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "lobed", "spike"): ("Lamiaceae", "Coleus amboinicus"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Celosia argentea"),
    ("opposite", "zygomorphic", "5", "superior", "shrub", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Hygrophila auriculata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Ageratum conyzoides"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "fleshy", "solitary"): ("Portulacaceae", "Portulaca oleracea"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Anisomeles malabarica"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "linear", "raceme"): ("Scrophulariaceae", "Limnophila indica"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Amaranthus viridis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Sida cordifolia"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "lobed", "head"): ("Asteraceae", "Tridax procumbens"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "palmate", "umbel"): ("Apiaceae", "Hydrocotyle sibthorpioides"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "simple", "spike"): ("Lamiaceae", "Plectranthus barbatus"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "linear", "raceme"): ("Acanthaceae", "Rungia repens"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Gomphrena celosioides"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Blumea lacera"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "capsule", "palmate", "cyme"): ("Malvaceae", "Hibiscus hispidissimus"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Angelonia salicariifolia"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Hyptis suaveolens"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Barleria prionitis"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Emilia sonchifolia"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Anethum sowa"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "nutlet", "simple", "spike"): ("Lamiaceae", "Ocimum americanum"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "linear", "raceme"): ("Acanthaceae", "Justicia procumbens"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Aerva lanata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Vernonia cinerea"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Sida rhombifolia"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Stemodia verticillata"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Leonotis nepetifolia"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Elytraria acaulis"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Synedrella nodiflora"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Pimpinella tirupatiensis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "nutlet", "simple", "spike"): ("Lamiaceae", "Salvia involucrata"),
    ("opposite", "zygomorphic", "5", "superior", "shrub", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Strobilanthes kunthiana"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Digera muricata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "fleshy", "head"): ("Asteraceae", "Launaea sarmentosa"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Urena lobata"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Mecardonia procumbens"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Orthosiphon thymiflorus"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "cyme"): ("Acanthaceae", "Dicliptera paniculata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Chromolaena odorata"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Coriandrum sativum"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "simple", "spike"): ("Lamiaceae", "Vitex negundo"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Amaranthus spinosus"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Hemigraphis alternata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Phyllanthus niruri"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Hibiscus rosa-sinensis"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Lindernia crustacea"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Mentha arvensis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Thunbergia fragrans"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Cynoglossum zeylanicum"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Foeniculum vulgare"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "simple", "spike"): ("Lamiaceae", "Clerodendrum inerme"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Achyranthes aspera"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Asystasia gangetica"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Eupatorium ayapana"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Pavonia odorata"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Scoparia dulcis"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Pogostemon benghalensis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Rhinacanthus nasutus"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Sphaeranthus indicus"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Trachyspermum ammi"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "simple", "spike"): ("Lamiaceae", "Gmelina asiatica"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Cyathula prostrata"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Adhatoda vasica"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Wedelia chinensis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Sida acuta"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Antirrhinum majus"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Ocimum basilicum"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Peristrophe paniculata"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Xanthium strumarium"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Carum carvi"),
    ("opposite", "actinomorphic", "5", "superior", "shrub", "nutlet", "simple", "spike"): ("Lamiaceae", "Premna serratifolia"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "capsule", "simple", "panicle"): ("Amaranthaceae", "Pupalia lappacea"),
    ("opposite", "zygomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Blepharis maderaspatensis"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "toothed", "head"): ("Asteraceae", "Laggera aurita"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "cordate", "cyme"): ("Malvaceae", "Malvastrum coromandelianum"),
    ("alternate", "zygomorphic", "5", "superior", "herb", "capsule", "simple", "raceme"): ("Scrophulariaceae", "Verbascum thapsus"),
    ("alternate", "actinomorphic", "5", "superior", "herb", "nutlet", "ovate", "spike"): ("Lamiaceae", "Salvia officinalis"),
    ("opposite", "actinomorphic", "5", "superior", "herb", "capsule", "lanceolate", "raceme"): ("Acanthaceae", "Hygrophila schulli"),
    ("alternate", "actinomorphic", "5", "inferior", "herb", "achene", "serrate", "head"): ("Asteraceae", "Echinops echinatus"),
    ("alternate", "actinomorphic", "4", "inferior", "herb", "schizocarp", "pinnate", "umbel"): ("Apiaceae", "Cuminum cyminum")
}

# Expanded family descriptions with ethnobotanical uses and references (paraphrased to avoid copyright issues)
family_details = {
    "Lamiaceae": {
        "description": "A family of aromatic plants, including herbs and shrubs, commonly utilized in South Indian traditional medicine for their benefits in respiratory, digestive, and immune system support.",
        "ethnobotanical_uses": "Holy Basil (Ocimum tenuiflorum) is traditionally used in Ayurveda to alleviate colds, fevers, and stress-related conditions.",
        "reference": "General botanical knowledge, inspired by studies from the Foundation for Revitalisation of Local Health Traditions (FRLHT), 2010."
    },
    "Acanthaceae": {
        "description": "This family includes herbs and shrubs often used in Siddha medicine, particularly for treating respiratory issues.",
        "ethnobotanical_uses": "Malabar Nut (Justicia adhatoda) is frequently used to manage cough, asthma, and bronchitis in traditional practices.",
        "reference": "Medicinal Plants of India, FRLHT, 2004."
    },
    "Apiaceae": {
        "description": "Known as the carrot family, this group includes herbs with both culinary and medicinal applications in South Indian culture.",
        "ethnobotanical_uses": "Gotu Kola (Centella asiatica) is valued for its role in wound healing and improving cognitive functions.",
        "reference": "Indian Medicinal Plants, Vol. 2, Orient Longman, 1995."
    },
    "Nyctaginaceae": {
        "description": "Often called the four o’clock family, these herbs are used in Ayurvedic practices for addressing kidney and liver health concerns.",
        "ethnobotanical_uses": "Punarnava (Boerhavia diffusa) is known for its diuretic and anti-inflammatory effects in traditional medicine.",
        "reference": "Ethnobotanical studies, inspired by FRLHT documentation, 2010."
    },
    "Asteraceae": {
        "description": "The sunflower family consists of herbs widely applied in South Indian remedies for skin, hair, and liver health.",
        "ethnobotanical_uses": "Bhringaraj (Eclipta prostrata) is traditionally used to promote hair growth and support liver detoxification.",
        "reference": "The Plant List, www.theplantlist.org, accessed 2025."
    },
    "Amaranthaceae": {
        "description": "This family includes herbs used in South Indian medicine for their digestive and anti-inflammatory properties.",
        "ethnobotanical_uses": "Alternanthera sessilis is often employed to treat stomach issues and reduce inflammation.",
        "reference": "Ethnobotany of South India, FRLHT, 2010."
    },
    "Malvaceae": {
        "description": "Known as the mallow family, these plants are used in traditional South Indian medicine for pain relief and wound healing.",
        "ethnobotanical_uses": "Abutilon indicum is valued for its pain-relieving and anti-inflammatory properties in folk medicine.",
        "reference": "General botanical knowledge, inspired by FRLHT studies, 2008."
    },
    "Scrophulariaceae": {
        "description": "The figwort family includes herbs recognized for their cognitive and anti-inflammatory benefits in South Indian traditions.",
        "ethnobotanical_uses": "Brahmi (Bacopa monnieri) is used to enhance memory and reduce anxiety in traditional practices.",
        "reference": "Indian Medicinal Plants, Vol. 3, Orient Longman, 1996."
    },
    "Portulacaceae": {
        "description": "Known as the purslane family, these herbs are valued for their nutritional content and anti-inflammatory effects.",
        "ethnobotanical_uses": "Purslane (Portulaca oleracea) is used for its omega-3 fatty acids and anti-inflammatory properties in traditional diets.",
        "reference": "Medicinal Plants of South India, FRLHT, 2008."
    }
}

# Taxonomic hierarchy for each species
taxonomy_data = {
    "Ocimum tenuiflorum": {
        "Kingdom": "Plantae",
        "Division": "Magnoliophyta",
        "Class": "Magnoliopsida",
        "Order": "Lamiales",
        "Family": "Lamiaceae",
        "Genus": "Ocimum",
        "Species": "tenuiflorum",
        "description": "Aromatic herb with opposite, simple leaves, purple or green, often hairy; flowers in spikes, white to purplish."
    },
    "Justicia adhatoda": {
        "Kingdom": "Plantae",
        "Division": "Magnoliophyta",
        "Class": "Magnoliopsida",
        "Order": "Lamiales",
        "Family": "Acanthaceae",
        "Genus": "Justicia",
        "Species": "adhatoda",
        "description": "Shrub with opposite, lanceolate leaves; white zygomorphic flowers in racemes."
    },
    "Centella asiatica": {
        "Kingdom": "Plantae",
        "Division": "Magnoliophyta",
        "Class": "Magnoliopsida",
        "Order": "Apiales",
        "Family": "Apiaceae",
        "Genus": "Centella",
        "Species": "asiatica",
        "description": "Creeping herb with alternate, palmate leaves; small actinomorphic flowers in umbels."
    },
    "Boerhavia diffusa": {
        "Kingdom": "Plantae",
        "Division": "Magnoliophyta",
        "Class": "Magnoliopsida",
        "Order": "Caryophyllales",
        "Family": "Nyctaginaceae",
        "Genus": "Boerhavia",
        "Species": "diffusa",
        "description": "Prostrate herb with alternate, simple leaves; pink actinomorphic flowers in cymes."
    },
    "Eclipta prostrata": {
        "Kingdom": "Plantae",
        "Division": "Magnoliophyta",
        "Class": "Magnoliopsida",
        "Order": "Asterales",
        "Family": "Asteraceae",
        "Genus": "Eclipta",
        "Species": "prostrata",
        "description": "Prostrate herb with alternate, toothed leaves; white to yellow flower heads."
    }
    # Add taxonomy for all 100 species (abbreviated here for brevity)
}

# Prepare dataset for decision tree
data = []
labels = []
families = []
for traits, (family, species) in characteristics_to_species.items():
    data.append(traits)
    labels.append(species)
    families.append(family)

df = pd.DataFrame(data, columns=["leaf_arrangement", "flower_symmetry", "petal_number", "ovary_position", "habit", "fruit_type", "leaf_shape", "inflorescence_type"])
df["species"] = labels
df["family"] = families

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop(columns=["species", "family"]), columns=["leaf_arrangement", "flower_symmetry", "ovary_position", "habit", "fruit_type", "leaf_shape", "inflorescence_type"])
df_encoded["petal_number"] = df["petal_number"].astype(int)

# Train decision tree with cross-validation using non-stratified KFold
clf = DecisionTreeClassifier(random_state=42, min_samples_leaf=1)
clf.fit(df_encoded, df["species"])
# Use KFold without stratification since each class has only 1 sample
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, df_encoded, df["species"], cv=kf)
model_accuracy = np.mean(scores) * 100



# Sidebar navigation with session state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Welcome"

page = st.sidebar.selectbox(
    "Navigate",
    ["Welcome", "Classifier", "Contacts"],
    index=["Welcome", "Classifier", "Contacts"].index(st.session_state.selected_page),
    key="page_selector"
)

# Update session state when sidebar selection changes
st.session_state.selected_page = page

# Initialize session state for inputs and prediction
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "leaf_arrangement": "",
        "flower_symmetry": "",
        "petal_number": "",
        "ovary_position": "",
        "habit": "",
        "fruit_type": "",
        "leaf_shape": "",
        "inflorescence_type": ""
    }
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

def predict_species(inputs):
    try:
        print(f"predict_species inputs: {inputs}")
        input_data = pd.DataFrame([inputs])
        for col in df.columns:
            if col not in input_data.columns and col not in ["species", "family"]:
                input_data[col] = None if col != "petal_number" else 0
        input_encoded = pd.get_dummies(input_data, columns=["leaf_arrangement", "flower_symmetry", "ovary_position", "habit", "fruit_type", "leaf_shape", "inflorescence_type"])
        for col in df_encoded.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[df_encoded.columns]
        input_encoded["petal_number"] = pd.to_numeric(input_data["petal_number"], errors="coerce").fillna(0).astype(int)
        pred = clf.predict(input_encoded)
        proba = clf.predict_proba(input_encoded)
        confidence = np.max(proba)
        if confidence < 0.7:
            st.warning("Low confidence prediction. Results may be inaccurate.")
        print(f"Prediction: {pred[0]}, Confidence: {confidence:.2%}")
        return pred[0], confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        print(f"Prediction error: {str(e)}")
        return None, 0.0

def validate_inputs(inputs):
    valid_options = {
        "leaf_arrangement": ["alternate", "opposite", "whorled"],
        "flower_symmetry": ["actinomorphic", "zygomorphic"],
        "petal_number": ["3", "4", "5", "6"],
        "ovary_position": ["superior", "inferior"],
        "habit": ["herb", "shrub"],
        "fruit_type": ["capsule", "nutlet", "schizocarp", "achene", "berry"],
        "leaf_shape": ["simple", "palmate", "pinnate", "lobed", "cordate", "lanceolate", "ovate", "linear", "toothed", "serrate", "fleshy"],
        "inflorescence_type": ["spike", "raceme", "umbel", "cyme", "head", "panicle", "solitary"]
    }
    for key, value in inputs.items():
        if value not in valid_options[key]:
            return False, f"Invalid value for {key}: {value}"
    return True, ""

def generate_pdf_report(species, family, confidence, taxonomy, family_info, inputs):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750

    c.drawString(100, y, "PLANTIFY! Classification Report ")
    y -= 20
    c.drawString(100, y, f"Species: {species}")
    y -= 20
    c.drawString(100, y, f"Family: {family}")
    y -= 20
    c.drawString(100, y, f"Confidence: {confidence:.2%}")
    y -= 30

    c.drawString(100, y, "Taxonomic Hierarchy:")
    y -= 20
    for key, value in taxonomy.items():
        c.drawString(120, y, f"{key}: {value}")
        y -= 15
    y -= 10

    c.drawString(100, y, "Morphological Characteristics Used:")
    y -= 20
    for key, value in inputs.items():
        c.drawString(120, y, f"{key.replace('_', ' ').title()}: {value}")
        y -= 15
    y -= 10

    c.drawString(100, y, "Family Details:")
    y -= 20
    c.drawString(120, y, f"Description: {family_info['description']}")
    y -= 20
    c.drawString(120, y, f"Ethnobotanical Uses: {family_info['ethnobotanical_uses']}")
    y -= 20
    c.drawString(120, y, f"Reference: {family_info['reference']}")
    y -= 20

    c.drawString(100, y, "Disclaimer: This classification report is generated for educational purposes to aid in the study of South Indian medicinal herbs. It should not be used for professional field identification without consulting a trained botanist.")
    y -= 20
    c.drawString(100, y, "Acknowledgment: Information in this report is based on publicly available botanical studies with proper attribution to sources like FRLHT and The Plant List.")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Welcome Page
if page == "Welcome":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("🌿 PLANTIFY! ")
    st.markdown("""
    This app is designed to help you identify South Indian medicinal herbs using AI-driven classification. Explore the rich heritage of traditional healing systems like Ayurveda and Siddha, and learn about the cultural and ecological significance of these plants.
    
    **Get started by clicking the button below!**
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Plant Classification"):
            st.session_state.selected_page = "Classifier"
            st.rerun()
    with col2:
        if st.button("Contacts"):
            st.session_state.selected_page = "Contacts"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Classifier Page
elif page == "Classifier":
    st.title("🌿 PLANTIFY!")
    st.markdown("""
    This app only developed for identify the south indian herbaceous plant which is native in india by AI agent.
    """, unsafe_allow_html=True)
    st.markdown("Select all characteristics below to classify the plant.")
    st.markdown("*Disclaimer*: This app is intended as an educational tool to assist in learning about South Indian medicinal herbs and their classification. It is not designed for professional field identification and should be used alongside formal botanical education or expert guidance.")
   
    # Input fields with slide-in animation
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.subheader("Plant Characteristics")
    st.session_state.inputs["leaf_arrangement"] = st.selectbox(
        "Leaf arrangement", 
        ["", "alternate", "opposite", "whorled"], 
        index=["", "alternate", "opposite", "whorled"].index(st.session_state.inputs["leaf_arrangement"]), 
        key="leaf"
    )
    st.session_state.inputs["flower_symmetry"] = st.selectbox(
        "Flower symmetry", 
        ["", "actinomorphic", "zygomorphic"], 
        index=["", "actinomorphic", "zygomorphic"].index(st.session_state.inputs["flower_symmetry"]), 
        key="symmetry"
    )
    st.session_state.inputs["petal_number"] = st.selectbox(
        "Number of petals", 
        ["", "3", "4", "5", "6"], 
        index=["", "3", "4", "5", "6"].index(st.session_state.inputs["petal_number"]), 
        key="petals"
    )
    st.session_state.inputs["ovary_position"] = st.selectbox(
        "Ovary position", 
        ["", "superior", "inferior"], 
        index=["", "superior", "inferior"].index(st.session_state.inputs["ovary_position"]), 
        key="ovary"
    )
    st.session_state.inputs["habit"] = st.selectbox(
        "Habit", 
        ["", "herb", "shrub"], 
        index=["", "herb", "shrub"].index(st.session_state.inputs["habit"]), 
        key="habit"
    )
    st.session_state.inputs["fruit_type"] = st.selectbox(
        "Fruit type", 
        ["", "capsule", "nutlet", "schizocarp", "achene", "berry"], 
        index=["", "capsule", "nutlet", "schizocarp", "achene", "berry"].index(st.session_state.inputs["fruit_type"]), 
        key="fruit"
    )
    st.session_state.inputs["leaf_shape"] = st.selectbox(
        "Leaf shape", 
        ["", "simple", "palmate", "pinnate", "lobed", "cordate", "lanceolate", "ovate", "linear", "toothed", "serrate", "fleshy"], 
        index=["", "simple", "palmate", "pinnate", "lobed", "cordate", "lanceolate", "ovate", "linear", "toothed", "serrate", "fleshy"].index(st.session_state.inputs["leaf_shape"]), 
        key="leaf_shape"
    )
    st.session_state.inputs["inflorescence_type"] = st.selectbox(
        "Inflorescence type", 
        ["", "spike", "raceme", "umbel", "cyme", "head", "panicle", "solitary"], 
        index=["", "spike", "raceme", "umbel", "cyme", "head", "panicle", "solitary"].index(st.session_state.inputs["inflorescence_type"]), 
        key="inflorescence"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Validate inputs for button enablement
    all_filled = all(value and value != "" for value in st.session_state.inputs.values())
    print(f"Inputs: {st.session_state.inputs}, All filled: {all_filled}")

    # Classify button
    if st.button("Classify", disabled=not all_filled):
        try:
            print(f"Classify button clicked with inputs: {st.session_state.inputs}")
            is_valid, error_msg = validate_inputs(st.session_state.inputs)
            if not is_valid:
                st.error(f"Validation error: {error_msg}")
                print(f"Validation error: {error_msg}")
            else:
                prediction, confidence = predict_species(st.session_state.inputs)
                if prediction:
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.success("Classification completed successfully!")
                else:
                    st.error("Prediction failed. Please check inputs and try again.")
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            print(f"Classification error: {str(e)}")

    # Display prediction if available
    if st.session_state.prediction:
        species = st.session_state.prediction
        family = df[df["species"] == species]["family"].iloc[0]
        taxonomy = taxonomy_data.get(species, {
            "Kingdom": "Plantae",
            "Division": "Magnoliophyta",
            "Class": "Magnoliopsida",
            "Order": "Unknown",
            "Family": family,
            "Genus": species.split()[0],
            "Species": species.split()[1] if len(species.split()) > 1 else species,
           
        })
        family_info = family_details.get(family, {
        
            "ethnobotanical_uses": "Ethnobotanical uses not documented.",
            "reference": "Not available."
        })
        st.success("🌸 Classification Result")
        st.markdown(
            f"""
            **Predicted Species**: <span style='color:blue'>{species}</span>  
            **Family**: <span style='color:green'>{family}</span>  
            **Confidence**: <span style='color:purple'>{st.session_state.confidence:.2%}</span>  
            ### Taxonomic Hierarchy
            - **Kingdom**: {taxonomy['Kingdom']}  
            - **Division**: {taxonomy['Division']}  
            - **Class**: {taxonomy['Class']}  
            - **Order**: {taxonomy['Order']}  
            - **Family**: {taxonomy['Family']}  
            - **Genus**: {taxonomy['Genus']}  
            - **Species**: {taxonomy['Species']}  
            
            ### Family Details
            - **Description**: {family_info['description']}  
            - **Ethnobotanical Uses**: {family_info['ethnobotanical_uses']}  
            - **Reference**: {family_info['reference']}  
            """,
            unsafe_allow_html=True
        )
        # Download PDF report
        pdf_buffer = generate_pdf_report(species, family, st.session_state.confidence, taxonomy, family_info, st.session_state.inputs)
        st.download_button(
            label="Download Classification Report (PDF)",
            data=pdf_buffer,
            file_name=f"{species}_classification_report.pdf",
            mime="application/pdf"
        )
        # Links to botanical databases
        if st.button("Reset Classification"):
            st.session_state.inputs = {
                "leaf_arrangement": "",
                "flower_symmetry": "",
                "petal_number": "",
                "ovary_position": "",
                "habit": "",
                "fruit_type": "",
                "leaf_shape": "",
                "inflorescence_type": ""
            }
            st.session_state.prediction = None
            st.session_state.confidence = 0.0
            st.rerun()

    st.markdown("---")

# Contacts Page
elif page == "Contacts":
    st.title("📇 Contact Management")
    st.markdown("Save and contact herbalists or experts.")

    # Contact form
    with st.form(key="contact_form"):
        contact_name = st.text_input("Herbalist Name")
        contact_phone = st.text_input("Phone Number")
        contact_email = st.text_input("Email")
        col1, col2, col3 = st.columns(3)
        with col1:
            save_button = st.form_submit_button("Save Contact")
        with col2:
            email_button = st.form_submit_button("Send Email")
        with col3:
            call_button = st.form_submit_button("Make Call")

    # Contact logic
    if save_button:
        if contact_name and (contact_phone or contact_email):
            with open("contacts.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if os.path.getsize("contacts.csv") == 0:
                    writer.writerow(["Name", "Phone", "Email"])
                writer.writerow([contact_name, contact_phone, contact_email])
            st.success("Contact saved successfully!")
        else:
            st.error("Please enter a name and at least one contact detail.")

    if email_button and contact_email:
        webbrowser.open(f"mailto:{contact_email}?subject=Herb%20Inquiry&body=Hello,%20I%20have%20a%20question%20about%20medicinal%20herbs.")
        st.success("Email client opened!")
    elif email_button:
        st.error("Please enter an email address.")

    if call_button and contact_phone:
        webbrowser.open(f"tel:{contact_phone}")
        st.success("Dialer opened!")
    elif call_button:
        st.error("Please enter a phone number.")

    # Display saved contacts
    if os.path.exists("contacts.csv"):
        with open("contacts.csv", "r") as f:
            contacts = list(csv.reader(f))
            if len(contacts) > 1:
                st.markdown("### Saved Contacts")
                for contact in contacts[1:]:
                    st.markdown(f"**{contact[0]}**: Phone: {contact[1] or 'N/A'}, Email: {contact[2] or 'N/A'}")

    st.markdown("---")

# Common footer for all pages
st.markdown("### The Importance of South Indian Medicinal Herbs")
st.markdown("""
South Indian medicinal herbs have been a cornerstone of traditional healing systems like Ayurveda and Siddha for centuries. Plants like Holy Basil (Ocimum tenuiflorum) and Malabar Nut (Justicia adhatoda) offer remedies for various ailments, from respiratory issues to stress relief. Beyond their medicinal value, these herbs play a vital role in preserving biodiversity and cultural heritage in South India, supporting ecosystems and traditional knowledge passed down through generations.
""")
st.markdown("#### Reflections on the Value of Plants")
st.markdown("""
- *“In every leaf of a medicinal herb, there lies a remedy—a gift from nature to heal humanity.”*  
  — Inspired by Ayurvedic wisdom  
- *“Plants are the lungs of the earth; preserving them ensures we breathe a future of health and harmony.”*  
  — Adapted from global ecological insights  
- *“The healing power of nature is boundless; every herb tells a story of life and resilience.”*  
  — A reflection on South Indian ethnobotany
""")

st.markdown("Built with Streamlit | Data source: South Indian medicinal plant studies")
st.markdown("Thank you for using south indian medicinal herb classifier")
