# Early Lung Cancer Detection System

A Django-based web application that uses AI-powered deep learning models to analyze CT scan images for early lung cancer detection. The system employs an ensemble of three pre-trained models (ResNet50, DenseNet121, and Enhanced CNN) to provide accurate predictions with confidence scores.

## 🚀 Features

- **AI-Powered Analysis**: Uses ensemble of 3 deep learning models for accurate predictions
- **User Management**: Collects patient information and generates unique report IDs
- **Report Tracking**: Search and retrieve analysis reports using unique IDs
- **Professional Reports**: Print-friendly medical reports with patient details
- **Real-time Predictions**: Instant analysis with confidence scores and visualizations
- **Responsive Design**: Modern Bootstrap 5 interface that works on all devices

## 🏥 Medical Classifications

The system classifies CT scan images into three categories:
- **Benign**: Non-cancerous tissue
- **Malignant**: Cancerous tissue requiring immediate attention
- **Normal**: Healthy lung tissue

## 🛠️ Technology Stack

- **Backend**: Django 5.2, Python 3.12
- **AI/ML**: TensorFlow 2.19, Keras 3.11
- **Frontend**: Bootstrap 5, Chart.js, Font Awesome
- **Database**: SQLite (development), PostgreSQL (production ready)
- **Image Processing**: PIL (Python Imaging Library)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Prasiddha9999/Early_Cancer_Prediction_System.git
cd Early_Cancer_Prediction_System
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

Place the following model files in the project root directory:
- `best_ResNet50.keras`
- `best_DenseNet121.keras`
- `best_EnhancedCNN.keras`

*Note: Model files are not included in the repository due to size constraints. Contact the repository owner for access to the trained models.*

### 5. Database Setup

```bash
python manage.py migrate
```

### 6. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 7. Run the Development Server

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## 📱 Usage

### For Patients/Users:

1. **Upload Analysis**:
   - Enter your full name
   - Upload a CT scan image (JPG, PNG, BMP, TIFF)
   - Click "Analyze Image"

2. **View Results**:
   - Review the AI analysis with confidence scores
   - Note your unique Report ID for future reference
   - Print the report if needed

3. **Retrieve Reports**:
   - Use the "Find Your Report" section
   - Enter your Report ID to access previous results

### For Developers:

1. **Admin Interface**: Access `/admin/` to manage data
2. **API Endpoints**: RESTful endpoints available for integration
3. **Model Management**: Easy to add/update AI models

## 🏗️ Project Structure

```
Early_Cancer_Prediction_System/
├── lung_cancer_detection/          # Django project settings
├── prediction/                     # Main application
│   ├── models.py                  # Database models
│   ├── views.py                   # Application logic
│   ├── forms.py                   # Form definitions
│   ├── ml_utils.py                # AI model utilities
│   ├── templates/                 # HTML templates
│   └── static/                    # CSS, JS, images
├── media/                         # Uploaded images
├── static/                        # Static files
├── best_*.keras                   # AI model files
├── requirements.txt               # Python dependencies
└── manage.py                      # Django management script
```

## 🔬 AI Model Details

### Ensemble Architecture:
- **ResNet50**: Deep residual network for feature extraction
- **DenseNet121**: Dense connections for gradient flow
- **Enhanced CNN**: Custom architecture optimized for medical imaging

### Performance Metrics:
- High accuracy on medical imaging datasets
- Robust confidence scoring
- Fast inference time (<2 seconds per image)

## 🚀 Deployment

### Production Deployment:

1. **Environment Variables**:
   ```bash
   export DEBUG=False
   export SECRET_KEY=your-secret-key
   export DATABASE_URL=your-database-url
   ```

2. **Static Files**:
   ```bash
   python manage.py collectstatic
   ```

3. **Database Migration**:
   ```bash
   python manage.py migrate
   ```

### Recommended Hosting:
- **Heroku**: Easy deployment with PostgreSQL
- **AWS EC2**: Full control with load balancing
- **DigitalOcean**: Cost-effective with managed databases

## ⚠️ Important Disclaimers

- **Medical Use**: This system is for research and educational purposes only
- **Not a Substitute**: Should not replace professional medical diagnosis
- **Accuracy**: While highly accurate, results should be verified by medical professionals
- **Privacy**: Ensure HIPAA compliance for medical data handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Prasiddha** - *Initial work* - [Prasiddha9999](https://github.com/Prasiddha9999)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Django community for the web framework
- Medical imaging research community for datasets and methodologies

## 📞 Support

For support, message in regmicode.com or create an issue in the GitHub repository.

---

**⚡ Quick Start**: `git clone → pip install -r requirements.txt → python manage.py migrate → python manage.py runserver`
