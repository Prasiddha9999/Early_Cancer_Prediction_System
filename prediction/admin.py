from django.contrib import admin
from .models import PredictionResult

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'predicted_class', 'confidence', 'best_model', 'upload_timestamp']
    list_filter = ['predicted_class', 'best_model', 'upload_timestamp']
    search_fields = ['original_filename', 'predicted_class']
    readonly_fields = ['upload_timestamp']
    ordering = ['-upload_timestamp']
