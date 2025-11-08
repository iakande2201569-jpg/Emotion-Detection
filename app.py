from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
import os

app = Flask(__name__)

# --- Ensure folders exist ---
if not os.path.exists('static'):
    os.makedirs('static')

if not os.path.exists('database'):
    os.makedirs('database')

# --- Initialize database ---
def init_db():
    conn = sqlite3.connect('database/data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            emotion TEXT,
            mode TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database is ready.")


init_db()

# --- Load model ---
model = load_model('face_emotionModel.h5')
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    user_name = request.form['username']
    mode = request.form['mode']
    img_file = request.files['imagefile']

    # Save image
    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(48,48), color_mode='grayscale')
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict emotion
    pred = model.predict(img)
    emotion = emotions[np.argmax(pred)]

    # Save details to database
    conn = sqlite3.connect('database/data.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (name, filename, emotion, mode) VALUES (?, ?, ?, ?)",
        (user_name, img_path, emotion, mode)
    )
    conn.commit()
    conn.close()

    # Return results
    return render_template('index.html',
                           prediction=emotion,
                           image=img_path,
                           name=user_name,
                           mode=mode)


@app.route('/clear_db')
def clear_db():
    conn = sqlite3.connect('database/data.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return "✅ All data has been cleared from the database."

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
