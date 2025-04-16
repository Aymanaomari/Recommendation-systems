from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
from typing import  Optional

app = Flask(__name__)

# Load model and mappings with validation
def load_model():
    try:
        # Load SVD components
        U = np.load('./model/svd_U.npy')
        sigma = np.load('./model/svd_sigma.npy')
        Vt = np.load('./model/svd_Vt.npy')
        
        # Load ratings matrix
        with open('./model/final_ratings_matrix.pkl', 'rb') as f:
            final_ratings_matrix = pickle.load(f)
        
        # Load ID mappings with case normalization
        with open('./model/user_to_index.pkl', 'rb') as f:
            user_to_index = {str(k).strip(): v for k, v in pickle.load(f).items()}
        
        with open('./model/index_to_user.pkl', 'rb') as f:
            index_to_user = pickle.load(f)
        
        assert U.shape[0] == final_ratings_matrix.shape[0], "User dimension mismatch!"
        assert Vt.shape[1] == final_ratings_matrix.shape[1], "Item dimension mismatch!"
        
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        preds_matrix = pd.DataFrame(abs(all_user_predicted_ratings), 
                                  columns=final_ratings_matrix.columns)
        
        print(f"✅ Model loaded successfully with {len(user_to_index)} users and {final_ratings_matrix.shape[1]} products")
        return final_ratings_matrix, preds_matrix, user_to_index, index_to_user
    
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise

try:
    final_ratings_matrix, preds_matrix, user_to_index, index_to_user = load_model()
except Exception as e:
    print(f"Failed to initialize model: {e}")
    exit(1)

def find_user_index(user_id: str) -> Optional[int]:
    """Flexible user lookup with case and whitespace handling"""
    clean_id = str(user_id).strip().lower()  
    for uid, idx in user_to_index.items():
        if str(uid).lower() == clean_id:  
            return idx
    return None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "user_count": len(user_to_index),
        "product_count": final_ratings_matrix.shape[1]
    })

@app.route('/api/users', methods=['GET'])
def list_users():
    page = int(request.args.get('page', 1))
    per_page = 50
    start = (page - 1) * per_page
    users = list(user_to_index.keys())[start:start + per_page]
    return jsonify({
        "users": users,
        "page": page,
        "total_users": len(user_to_index)
    })


@app.route('/api/recommend', methods=['GET'])
def recommend_api():
    try:
        user_id = request.args.get('user_id', '').strip()
        limit = min(int(request.args.get('limit', 5)), 20)  # Max 20 recommendations
        
        if not user_id:
            return jsonify({'status': 'error', 'message': 'user_id is required'}), 400
        
        # Find user with flexible matching
        user_index = find_user_index(user_id)
        
        if user_index is None:

            similar = [uid for uid in user_to_index.keys() 
                      if user_id.lower() in uid.lower()][:5]
            return jsonify({
                'status': 'error',
                'message': f'User "{user_id}" not found',
                'suggestions': similar,
                'total_users': len(user_to_index)
            }), 404
            
        user_ratings = final_ratings_matrix.iloc[user_index]
        user_predictions = preds_matrix.iloc[user_index]
        
        unrated_mask = (user_ratings == 0)
        recommendations = user_predictions[unrated_mask]\
                         .sort_values(ascending=False)\
                         .head(limit)
        
        response = {
            'status': 'success',
            'user_id': index_to_user.get(user_index, user_id),
            'already_rated': int(sum(user_ratings > 0)),
            'recommendations': {
                str(prod_id): float(rating) 
                for prod_id, rating in recommendations.items()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
