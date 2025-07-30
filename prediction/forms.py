from django import forms
from .models import PredictionResult

class ImageUploadForm(forms.ModelForm):
    """Form for uploading CT scan images"""

    class Meta:
        model = PredictionResult
        fields = ['user_name', 'uploaded_image']
        widgets = {
            'user_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your full name',
                'required': True
            }),
            'uploaded_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'imageUpload'
            })
        }
        labels = {
            'user_name': 'Your Name',
            'uploaded_image': 'Select CT Scan Image'
        }
    
    def clean_uploaded_image(self):
        """Validate uploaded image"""
        image = self.cleaned_data.get('uploaded_image')
        
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file too large. Please select an image smaller than 10MB.")
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            file_extension = image.name.lower().split('.')[-1]
            if f'.{file_extension}' not in valid_extensions:
                raise forms.ValidationError("Invalid file format. Please upload a valid image file (JPG, PNG, BMP, TIFF).")
        
        return image


class ReportSearchForm(forms.Form):
    """Form for searching reports by ID"""
    report_id = forms.CharField(
        max_length=12,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Report ID (e.g., ABC123DEF456)',
            'style': 'text-transform: uppercase;'
        }),
        label='Report ID'
    )

    def clean_report_id(self):
        """Clean and validate report ID"""
        report_id = self.cleaned_data.get('report_id', '').upper().strip()

        if len(report_id) != 12:
            raise forms.ValidationError("Report ID must be exactly 12 characters long.")

        if not report_id.isalnum():
            raise forms.ValidationError("Report ID must contain only letters and numbers.")

        return report_id
