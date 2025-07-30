from django.db import models
from django.utils import timezone
import json
import uuid

class PredictionResult(models.Model):
    """Model to store prediction results"""

    # User information
    user_name = models.CharField(max_length=100, help_text="Name of the person requesting the analysis")
    report_id = models.CharField(max_length=12, unique=True, help_text="Unique ID for report retrieval")

    # Image information
    uploaded_image = models.ImageField(upload_to='uploads/')
    original_filename = models.CharField(max_length=255)
    upload_timestamp = models.DateTimeField(default=timezone.now)

    # Prediction results (stored as JSON)
    predictions_json = models.TextField()  # Store all model predictions as JSON

    # Best prediction summary
    best_model = models.CharField(max_length=50, blank=True, null=True)
    predicted_class = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)

    # Ensemble prediction
    ensemble_class = models.CharField(max_length=50, blank=True, null=True)
    ensemble_confidence = models.FloatField(blank=True, null=True)

    class Meta:
        ordering = ['-upload_timestamp']

    def save(self, *args, **kwargs):
        """Override save to generate unique report ID"""
        if not self.report_id:
            self.report_id = self.generate_unique_report_id()
        super().save(*args, **kwargs)

    def generate_unique_report_id(self):
        """Generate a unique 12-character report ID"""
        while True:
            # Generate a random 12-character ID using UUID
            report_id = str(uuid.uuid4()).replace('-', '')[:12].upper()
            # Check if this ID already exists
            if not PredictionResult.objects.filter(report_id=report_id).exists():
                return report_id

    def __str__(self):
        if self.user_name and self.report_id:
            return f"Report {self.report_id} - {self.user_name} ({self.predicted_class or 'Processing...'})"
        else:
            return f"Prediction for {self.original_filename} - Processing..."

    @property
    def predictions(self):
        """Get predictions as Python dict"""
        return json.loads(self.predictions_json)

    def set_predictions(self, predictions_dict):
        """Set predictions from Python dict"""
        self.predictions_json = json.dumps(predictions_dict)

    def get_formatted_timestamp(self):
        """Get formatted timestamp"""
        return self.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S")
