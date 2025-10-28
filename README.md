echo "Flask==3.0.3\ngunicorn==23.0.0" > requirements.txt
echo "web: gunicorn app:app" > Procfile
git add .
git commit -m "Deploy"
git push
