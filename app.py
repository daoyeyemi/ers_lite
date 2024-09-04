from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, flash, session
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import sqlite3
import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
# from sklearn.linear_model import LinearRegression
# import base64
import os
from database_config import init_db, authenticate_user, create_user


app = Flask(__name__)

app.secret_key = os.urandom(24)

model_path = 'model/model.h5'
face_cascade_path = 'haar_cascade_classifier/haarcascade_frontalface_default.xml'

model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(face_cascade_path)

latest_emotion = { "emotion" : None, "timestamp" : None }

emotion_history = []

def process_emotion(label):
    global latest_emotion, emotion_history
    
    timestamp = datetime.datetime.now()
    
    formatted_timestamp = timestamp.strftime('%B %-d, %Y %-I:%M %p').lstrip('0').replace(' 0', ' ')

    formatted_timestamp = formatted_timestamp.replace(' 0', ' ')
    
    latest_emotion = {"emotion": label, "timestamp": formatted_timestamp}
    
    # Add the latest emotion to the history
    emotion_history.append(latest_emotion)
    
    # Keep only the last ten emotions
    if len(emotion_history) > 10:
        emotion_history.pop(0)
        
    print("printing all emotions in emotion history: ")
    
    for emotion in emotion_history:
        print(emotion)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_emotion_history', methods=['POST'])
def get_emotion_history():
    # Detect the emotion and get the label
    # Assuming `label` is the detected emotion
    # Replace with your actual prediction logic
    return jsonify(emotion_history)

# webcam feed route
@app.route('/video_feed', methods=['GET'])
def video_feed():
   
    def gen_frames():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 470))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                standardized_face = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = standardized_face.astype('float') / 255.0
                roi = np.expand_dims(img_to_array(roi), axis=0)
                prediction = model.predict(roi)[0]
                label = emotion_labels[np.argmax(prediction)]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                process_emotion(label)
                
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/history')
# def show_history():
    
#     username = session.get('username')
    
#     conn = sqlite3.connect('emotion_recognition_system_database.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10", (username,))
#     rows = c.fetchall()
#     conn.close()

#     history_data = []
#     for row in rows:
#         id, face_id, username, user_id, emotion, face_image_data, timestamp = row
#         timestamp_dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
#         formatted_timestamp = timestamp_dt.strftime('%B %-d, %Y - %-I:%M %p')
#         if face_image_data:
#             face_image = base64.b64encode(face_image_data).decode('utf-8')
#             history_data.append({
#                 'face_id': face_id,
#                 'emotion': emotion,
#                 'timestamp': formatted_timestamp,
#                 'face_image': face_image
#             })
#     return render_template('history.html', history=history_data)

# other routes for distribution, emotion over time, emotion by user, prediction, etc.
# @app.route('/plot_emotion_distribution')
# def plot_emotion_distribution():
#     conn = sqlite3.connect('emotion_recognition_system_database.db')
#     c = conn.cursor()
#     c.execute("SELECT emotion, COUNT(*) FROM predictions GROUP BY emotion")
#     data = c.fetchall()
#     conn.close()
    
#     emotions = [row[0] for row in data]
#     counts = [row[1] for row in data]
    
#     # create a Plotly pie chart
#     fig = px.pie(names=emotions, values=counts, title="Emotion Distribution")
    
#     # convert Plotly figure to HTML
#     plot_html = pio.to_html(fig, full_html=False)
    
#     return render_template('plot.html', plot_html=plot_html, title='Emotion Distribution')

# @app.route('/plot_emotion_over_time')
# def plot_emotion_over_time():
#     conn = sqlite3.connect('emotion_recognition_system_database.db')
#     c = conn.cursor()
#     c.execute("SELECT timestamp, emotion FROM predictions")
#     data = c.fetchall()
#     conn.close()
    
#     df = pd.DataFrame(data, columns=['timestamp', 'emotion'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['count'] = 1
    
#     df_grouped = df.groupby(['timestamp', 'emotion']).count().reset_index()
    
#     fig = px.line(df_grouped, x='timestamp', y='count', color='emotion', title="emotion over time", labels={'count' : 'count'})
        
#     # convert Plotly figure to HTML
#     plot_html = pio.to_html(fig, full_html=False)
    
#     return render_template('plot.html', plot_html=plot_html, title='emotions over time')

# @app.route('/plot_emotion_by_user')
# def plot_emotion_by_user():
#     conn = sqlite3.connect('emotion_recognition_system_database.db')
#     c = conn.cursor()
#     c.execute("SELECT username, emotion, COUNT(*) FROM predictions GROUP BY username, emotion")
#     data = c.fetchall()
#     conn.close()
    
#     df = pd.DataFrame(data, columns=['username', 'emotion', 'count'])
#     df = df.pivot(index='username', columns='emotion', values='count').fillna(0)
    
#     # Create a Plotly stacked bar chart
#     fig = px.bar(df, x=df.index, y=df.columns, title="Emotion by User", labels={'value': 'Count'})
    
#     # Convert Plotly figure to HTML
#     plot_html = pio.to_html(fig, full_html=False)
    
#     return render_template('plot.html', plot_html=plot_html, title='Emotions by User')

# @app.route('/predict_emotion_levels')
# def predict_emotion_levels():
#     conn = sqlite3.connect('emotion_recognition_system_database.db')
#     query = "SELECT timestamp, emotion FROM predictions"
#     df = pd.read_sql_query(query, conn)
#     conn.close()

#     # Convert timestamp to datetime and extract the hour
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['hour'] = df['timestamp'].dt.hour
    
#     emotions_of_interest = ['angry', 'disgust', 'fear', 'sad', 'surprise']
#     filtered_df = df[df['emotion'].isin(emotions_of_interest)]
    
#     if filtered_df.empty:
#         return "No records found for the specified emotions."
    
#     # Aggregate counts by hour and emotion
#     emotion_counts = filtered_df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)

#     # Initialize the plotly figure
#     fig = go.Figure()

#     # Loop through each emotion of interest
#     for emotion in emotions_of_interest:
#         if emotion in emotion_counts:
#             emotion_data = emotion_counts[emotion].reset_index(name='count')
#             fig.add_trace(go.Scatter(x=emotion_data['hour'], y=emotion_data['count'], mode='lines+markers', name=f'Observed {emotion.capitalize()} Counts'))

#             # Perform regression analysis
#             X = emotion_data[['hour']]
#             y = emotion_data['count']
#             model = LinearRegression()
#             model.fit(X, y)

#             # Predict counts for each hour
#             hours = np.arange(0, 24).reshape(-1, 1)
#             predictions = model.predict(hours)

#             # Add regression line to the plot
#             fig.add_trace(go.Scatter(x=hours.flatten(), y=predictions, mode='lines', name=f'Predicted {emotion.capitalize()} Counts', line=dict(dash='dash')))

#     # Customize the layout
#     fig.update_layout(
#         title='Regression Analysis: Predicted Emotion Levels by Hour',
#         xaxis_title='hour of day',
#         yaxis_title='counts',
#         legend_title='emotions',
#         template='plotly_white'
#     )

#     # Convert Plotly figure to HTML
#     plot_html = pio.to_html(fig, full_html=False)
    
#     return render_template('plot.html', plot_html=plot_html, title='Emotion Level Predictions')

if __name__ == "__main__":
    # init_db()
    app.run(debug=True)