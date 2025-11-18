from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pickle
import os

app = Flask(__name__)

FEATURES_PATH = 'feature_names.pkl'
MODEL_FILES = {
    'Random Forest': 'Random_Forest_Model.pkl',
    'Logistic Regression': 'Logistic_Regression_Model.pkl',
    'Naive Bayes': 'Naive_Bayes_Model.pkl',
    'Gradient Boosting': 'Gradient_Boosting_Model.pkl'
}


def load_feature_names():
    """Load feature names if available"""
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def smiles_to_features(smiles):
    """Convert SMILES to molecular features - must match training features exactly"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate Morgan fingerprints (2048 features) - matches training
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprint = list(fp)
        
        # Calculate molecular descriptors (10 features) - matches training exactly
        descriptors = [
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.MolLogP(mol),
            Descriptors.RingCount(mol)
        ]
        
        # Calculate Molecular Weight and AlogP separately (2 features)
        molecular_weight = Descriptors.MolWt(mol)
        alogp = Descriptors.MolLogP(mol)  # Using MolLogP as AlogP approximation
        
        # Combine: fingerprint (2048) + descriptors (10) + Molecular Weight (1) + AlogP (1) = 2060
        feature_array = np.array(fingerprint + descriptors + [molecular_weight, alogp])
        return feature_array.reshape(1, -1)
    
    except Exception as e:
        print(f"Error processing SMILES: {e}")
        return None

def get_molecular_properties(smiles):
    """Extract molecular properties for display"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'molecular_weight': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'h_bond_donors': int(Descriptors.NumHDonors(mol)),
            'h_bond_acceptors': int(Descriptors.NumHAcceptors(mol)),
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'rotatable_bonds': int(Descriptors.NumRotatableBonds(mol)),
            'aromatic_rings': int(Descriptors.NumAromaticRings(mol)),
            'total_rings': int(Descriptors.RingCount(mol))
        }
    except Exception as e:
        print(f"Error calculating molecular properties: {e}")
        return None

def load_all_models():
    """Load all trained models once at startup"""
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            print(f"Warning: {path} not found")
    return models


FEATURE_NAMES = load_feature_names()
MODELS = load_all_models()


@app.route('/')
def home():
    return render_template('index.html', model_names=list(MODELS.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        smiles = data.get('smiles', '')
        model_name = data.get('model_name', 'Random Forest')
        
        if not smiles:
            return jsonify({'error': 'Please provide a SMILES string'}), 400
        
        # Convert SMILES to features
        features = smiles_to_features(smiles)
        
        if features is None:
            return jsonify({'error': 'Invalid SMILES string'}), 400
        
        # Get molecular properties for display
        molecular_properties = get_molecular_properties(smiles)
        
        if molecular_properties is None:
            return jsonify({'error': 'Failed to calculate molecular properties'}), 400
        
        if model_name not in MODELS:
            return jsonify({'error': f'Model {model_name} not found'}), 400

        model_obj = MODELS[model_name]
        
        try:
            # Handle NaN values for models that need it
            if model_name in ['Logistic Regression', 'Naive Bayes', 'Gradient Boosting']:
                if FEATURE_NAMES and len(FEATURE_NAMES) == features.shape[1]:
                    feature_df = pd.DataFrame(features, columns=FEATURE_NAMES)
                else:
                    feature_df = pd.DataFrame(features)
                features_processed = feature_df.fillna(feature_df.mean()).values
            else:
                features_processed = features
            
            pred = model_obj.predict(features_processed)[0]
            proba = model_obj.predict_proba(features_processed)[0]
            
            return jsonify({
                'smiles': smiles,
                'model_name': model_name,
                'prediction': int(pred),
                'probabilities': {
                    'class_0': float(proba[0]),
                    'class_1': float(proba[1])
                },
                'confidence': float(max(proba[0], proba[1])),
                'features': molecular_properties
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
