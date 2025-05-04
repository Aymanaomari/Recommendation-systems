from app import app
from app.recommendations import collaborative_filtering_recommendations, svd_recommendations, rank_based_recommendations,item_based_recommendations
from flask import jsonify, request

@app.route('/collaborative_filtering', methods=['GET'])
def collaborative_filtering():
    user_id = request.args.get('user_id')
    num_of_products = int(request.args.get('num_of_products', 5))
    recommended_products = collaborative_filtering_recommendations(user_id, num_of_products)
    return jsonify(recommended_products)

@app.route('/svd_recommendations', methods=['GET'])
def svd_recommendations_endpoint():
    user_id = request.args.get('user_id')
    num_of_products = int(request.args.get('num_of_products', 5))
    recommended_products = svd_recommendations(user_id, num_of_products)
    return jsonify(recommended_products)

@app.route('/rank_based_recommendations', methods=['GET'])
def rank_based_recommendations_endpoint():
    n = int(request.args.get('num_of_products', 10))
    recommended_products = rank_based_recommendations(n)
    return jsonify(recommended_products)


@app.route("/item_based_recommendations",methods=["GET"])
def item_based_recommendations_endpoint():
    user_id = request.args.get('user_id')
    num_of_products = int(request.args.get('num_of_products', 5))
    recommended_products = item_based_recommendations(user_id,num_of_products,10)
    return jsonify(recommended_products)
