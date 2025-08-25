# 🖼️ AI-Driven Image Super-Resolution & Neural Style Transfer Web App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Django](https://img.shields.io/badge/Django-Backend-green?logo=django)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This project is a **web-based application** that integrates two cutting-edge computer vision techniques: **Image Super-Resolution** and **Neural Style Transfer**.  
Users can enhance low-quality images into high-resolution outputs and generate artistic images by blending the content of one image with the style of another.

---

## 🚀 Features

### 🔹 Image Super-Resolution
- Upload low-resolution images and upscale them using a trained **U-Net** model.  
- Model trained with **supervised learning** on paired LR-HR datasets.  
- **Data augmentation**: cropping, flipping, scaling for better generalization.  
- **Optimization**: Mean Squared Error (MSE) loss + Adam optimizer.  
- **Architecture**: U-Net with encoder-decoder + skip connections for sharp detail reconstruction.

### 🔹 Neural Style Transfer
- Upload a **content image** + a **style image**.  
- Uses pre-trained **VGG19** model from PyTorch for deep feature extraction.  
- Iterative optimization produces an image with **content preserved** but **style transformed**.  
- Loss function: **content loss + style loss (Gram matrices) + total variation loss**.  

### 🔹 Web Application
- **Frontend**: Clean UI built with **HTML, CSS, JavaScript**.  
- **Backend**: Powered by **Django** for user authentication, image processing, routing, and file handling.  
- Secure **registration/login system** using Django’s built-in authentication.  
- Users can **view and download** processed images.  

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Django (Python)  
- **Deep Learning Models**:
  - U-Net → Image Super-Resolution  
  - VGG19 → Neural Style Transfer  
- **Libraries**: PyTorch, NumPy, Pillow  

---

## 📂 Project Structure
```bash
project-root/
│── super_resolution/        # U-Net model, training & inference
│── style_transfer/          # Style transfer implementation
│── media/                   # Uploaded & processed images
│── static/                  # CSS, JS, frontend assets
│── templates/               # HTML templates
│── users/                   # Authentication module
│── views.py                 # Django views & routing
│── urls.py                  # URL configuration
│── models.py                # Django models
│── manage.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Database Migrations
```bash
python manage.py migrate
```

### 5️⃣ Start the Development Server
```bash
python manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000/** 🎉  

---

## 📸 Demo
👉 *(Add screenshots or GIFs here)*  

- **Super Resolution Example:**  
  *Low-res → High-res*  

- **Style Transfer Example:**  
  *Content + Style → Artistic Output*  

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).  

---

## 👨‍💻 Author
Developed by **[Your Name](https://github.com/your-username)** ✨  
Feel free to ⭐ the repo if you found it helpful!
