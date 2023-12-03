from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Konfigurasi SQLAlchemy untuk database MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/fitfans'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model untuk tabel pengguna
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)

with app.app_context():
    # Buat tabel pada database
    db.create_all()

# Endpoint untuk mendapatkan data dari tabel users
@app.route('/users', methods=['GET'])
def users_endpoint():
    try:
        user_id = request.args.get('user_id')
        user_email = request.args.get('user_email')

        if user_id and user_email:
            return jsonify({'error': 'Provide either user_id or user_email, not both.'}), 400
        elif user_id:
            user = User.query.get(user_id)
            if user:
                return jsonify({'user': {'id': user.id, 'full_name': user.full_name, 'age': user.age, 'weight': user.weight, 'height': user.height, 'gender': user.gender, 'email': user.email}}), 200
            else:
                return jsonify({'message': 'User not found'}), 404
        elif user_email:
            user = User.query.filter_by(email=user_email).first()
            if user:
                return jsonify({'user': {'id': user.id, 'full_name': user.full_name, 'age': user.age, 'weight': user.weight, 'height': user.height, 'gender': user.gender, 'email': user.email}}), 200
            else:
                return jsonify({'message': 'User not found'}), 404
        else:
            users = User.query.all()
            user_list = [{'id': user.id, 'full_name': user.full_name, 'age': user.age, 'weight': user.weight, 'height': user.height, 'gender': user.gender, 'email': user.email} for user in users]
            return jsonify({'users': user_list}), 200
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

# Endpoint untuk menambahkan pengguna baru
@app.route('/users', methods=['POST'])
def add_user():
    try:
        new_user_data = request.json

        if not new_user_data or not all(key in new_user_data for key in ['full_name', 'age', 'weight', 'height', 'gender', 'email']):
            return jsonify({'error': 'Bad Request - Invalid User Data'}), 400

        new_user = User(**new_user_data)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'message': 'User added successfully', 'user': {'id': new_user.id, 'full_name': new_user.full_name, 'age': new_user.age, 'weight': new_user.weight, 'height': new_user.height, 'gender': new_user.gender, 'email': new_user.email}}), 201
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

# Endpoint untuk mengedit pengguna
@app.route('/users/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    try:
        user = User.query.get(user_id)
        if user:
            updated_user_data = request.json

            if not updated_user_data or not all(key in updated_user_data for key in ['full_name', 'age', 'weight', 'height', 'gender', 'email']):
                return jsonify({'error': 'Bad Request - Invalid User Data'}), 400

            user.full_name = updated_user_data['full_name']
            user.age = updated_user_data['age']
            user.weight = updated_user_data['weight']
            user.height = updated_user_data['height']
            user.gender = updated_user_data['gender']
            user.email = updated_user_data['email']

            db.session.commit()

            return jsonify({'message': 'User updated successfully', 'user': {'id': user.id, 'full_name': user.full_name, 'age': user.age, 'weight': user.weight, 'height': user.height, 'gender': user.gender, 'email': user.email}}), 200
        else:
            return jsonify({'message': 'User not found'}), 404
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

# Endpoint untuk menghapus pengguna
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
            return jsonify({'message': 'User deleted successfully'}), 204
        else:
            return jsonify({'message': 'User not found'}), 404
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

# Load pre-trained model
model = tf.keras.models.load_model('Gym_Tools_Multi.h5')

# Nama kelas untuk prediksi
class_names = ['barbell', 'dumbell', 'gym-ball', 'kattle-ball', 'leg-press', 'punching-bag', 'roller-abs', 'statis-bicycle', 'step', 'treadmill']

# Fungsi untuk memprediksi kelas gambar
def predict_image_class(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    predictions = model.predict(img)
    max_probability = np.max(predictions)
    if max_probability < 0.5:
        return "Gambar tidak dikenali"
    else:
        predicted_class = np.argmax(predictions)
        return class_names[predicted_class]

# Endpoint untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Simpan gambar yang diunggah
    upload_path = 'uploads'
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    file_path = os.path.join(upload_path, file.filename)
    file.save(file_path)

    # Prediksi kelas untuk gambar yang diunggah
    predicted_class = predict_image_class(file_path)

    # Kembalikan hasil prediksi
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)