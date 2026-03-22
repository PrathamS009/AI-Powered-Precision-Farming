# ============================================================
# GOOGLE COLAB SETUP GUIDE FOR RICE LEAF DISEASE TRAINING
# Read this before running any notebook on Colab
# ============================================================

"""
STEP 1 — Open Google Colab
--------------------------
Go to: https://colab.research.google.com
Sign in with your Google account.
Click: File > New Notebook

STEP 2 — Enable GPU (VERY IMPORTANT, do this first)
----------------------------------------------------
Click: Runtime > Change runtime type
Under "Hardware Accelerator" select: T4 GPU
Click Save.

Without GPU, training will take 10-20x longer.

STEP 3 — Connect your Kaggle dataset
--------------------------------------
You need to upload your Kaggle API key so Colab can download your
private dataset directly.

3a. Get your Kaggle API key:
    - Go to https://www.kaggle.com
    - Click your profile picture > Settings
    - Scroll to "API" section
    - Click "Create New Token"
    - This downloads a file called kaggle.json

3b. In your Colab notebook, paste and run this cell:

    from google.colab import files
    files.upload()   # Upload your kaggle.json here

    import os
    os.makedirs('/root/.kaggle', exist_ok=True)

    # Move the uploaded file to the right place
    import shutil
    shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 600)

3c. Download your private dataset:

    !pip install kaggle
    !kaggle datasets download -d YOUR_KAGGLE_USERNAME/rice-leaf-disease-dataset
    !unzip rice-leaf-disease-dataset.zip -d /content/rice_data/

    NOTE: Replace YOUR_KAGGLE_USERNAME with your actual Kaggle username.
    Your dataset will be at:
        /content/rice_data/train/
        /content/rice_data/test/

    UPDATE the TRAIN_DIR and TEST_DIR in each training script:
        TRAIN_DIR = "/content/rice_data/train"
        TEST_DIR  = "/content/rice_data/test"

STEP 4 — Install required libraries
-------------------------------------
Run this cell at the start of every Colab session:

    !pip install tensorflow scikit-learn matplotlib seaborn pandas

STEP 5 — Upload your training script
--------------------------------------
Option A (easiest): Copy-paste the .py file content into a Colab code cell and run it.

Option B: Upload the .py file:
    from google.colab import files
    files.upload()   # Upload your .py file
    !python 1_efficientnet_b0_train.py

Option C: Mount Google Drive and run from there:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/drive/MyDrive/RiceLeaf/1_efficientnet_b0_train.py

STEP 6 — Download your trained models
---------------------------------------
After training, your models are saved at /kaggle/working/ on Colab.
Wait -- on Colab they save to /content/ or wherever OUTPUT_DIR points.

To download a model:
    from google.colab import files
    files.download('/content/efficientnet_b0/efficientnet_b0_final.keras')

Or save to Google Drive:
    import shutil
    shutil.copy('/content/efficientnet_b0/efficientnet_b0_final.keras',
                '/content/drive/MyDrive/RiceLeaf_Models/')

STEP 7 — Run each model one at a time
---------------------------------------
Do NOT run all 5 models in the same session.
Each model should be run in a fresh Colab session to avoid memory issues.

Recommended order (fastest to slowest):
    1. MobileNetV3Large   (~20-30 min on T4)
    2. EfficientNetB0     (~25-35 min on T4)
    3. EfficientNetB3     (~35-50 min on T4)
    4. ResNet50           (~40-55 min on T4)
    5. VGG16              (~50-70 min on T4)

IMPORTANT NOTES:
- Colab free tier disconnects after ~90 min of inactivity
- Keep the browser tab open while training
- Save models to Google Drive so you don't lose them if disconnected
- Free Colab gives ~12 hours of GPU per day

STEP 8 — After all models are trained
---------------------------------------
Run 6_evaluate_and_compare.py to generate the comparison charts.
Make sure all model .keras files are in the correct OUTPUT_DIR paths.
"""
