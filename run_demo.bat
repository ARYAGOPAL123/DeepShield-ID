@echo off
REM Demo runner for DeepShield-ID
if not exist .venv\Scripts\Activate.bat (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
)
call .venv\Scripts\Activate.bat
pip install -r requirements.txt

echo Generating a small synthetic ID dataset...
python data/synthesize_idcards.py --out data/processed --count 120 --fake-ratio 0.5

echo Training model for demo...
python model/train.py --data data/processed --epochs 5 --batch-size 16 --out model/weights.pth

echo Launching Streamlit UI...
py -3 -m streamlit run app/streamlit_app.py
pause
