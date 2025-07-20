# Tea Leaf Disease Classification

This project is a web application for classifying tea leaf diseases using a deep learning model (ResNet50). The backend is built with Flask, and the frontend is a simple HTML/JS interface for uploading images and viewing predictions.



## Project Structure

```

├── backend/
│   ├── app.py                
│   ├── requirements.txt      
│   ├── model/
│   │   ├── model.pth         
│   │   └── train_model.py    
│   ├── evaluate.py           
│   └── uploads/             
|
├── data/
|
├── frontend/
│   ├── index.html            
│   ├── script.js             
│   └── style.css   
|          
└── README.md                 
```

## Setup Instructions

### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare the Model
- upload the labeled dataset with folder name "data" in the root directory

- Train the model with the following command

```bash
cd model
python train_model.py
```
- `model.pth` will be created
- The script uses images from the `data/` folder.

### 3. Run the Backend Server
```bash
python app.py
```
- The server will start at [http://localhost:5000](http://localhost:5000).


### 4. Evaluate the Model (Optional)
- To generate evaluation metrics and plots:
```bash
python evaluate.py
```
- This will create `confusion_matrix.png` and `evaluation_report.txt` in the `backend` folder.

- Sample dataset Link: https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs


## Steps to Set Up Virtual Environment

### 1. Navigate to your project directory
```cmd
cd tea-leaf-disease-detection-model
```

### 2. Create a virtual environment
```cmd
python -m venv venv
```
This creates a new virtual environment named "venv" in your project directory.

### 3. Activate the virtual environment

On **Windows**:
```cmd
venv\Scripts\activate
```
On **macOS/Linux**:
```bash
source venv/bin/activate
```

On Windows, this activates the virtual environment. You'll see `venv` at the beginning of your command prompt.
- To deactive `venv` use
```
deactivate
```
