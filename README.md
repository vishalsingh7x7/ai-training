
# create vitual env
python3 -m venv venv

# activate it
source venv/bin/activate

# install requirements
pip install -r requirements.txt

# put api key in .env

# start mandala app
streamlit run mandala_app.py

# start face recognition app
streamlit run vision_poet_app.py     
