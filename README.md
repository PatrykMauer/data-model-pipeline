docker build -t flask-app:latest .

docker run -p 5000:5000 flask-app:latest


## Bash:
Model training

'''
curl -X POST http://localhost:5000/train_model
'''


Data processing:
'''
curl -X POST http://localhost:5000/webhook \
     -H "Content-Type: application/json" \
     -d '{"filename": "results_2024_05_11.xlsx"}'
'''