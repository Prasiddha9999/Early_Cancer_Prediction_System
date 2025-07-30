from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json

from .forms import ImageUploadForm, ReportSearchForm
from .models import PredictionResult
from .ml_utils import predict_image, get_available_models, get_model_info

def home(request):
    """Home page with image upload form and report search"""
    upload_form = ImageUploadForm()
    search_form = ReportSearchForm()
    recent_predictions = PredictionResult.objects.all()[:5]  # Show last 5 predictions
    available_models = get_available_models()
    model_info = get_model_info()

    context = {
        'upload_form': upload_form,
        'search_form': search_form,
        'recent_predictions': recent_predictions,
        'available_models': available_models,
        'total_models': len(available_models),
        'model_info': model_info
    }
    return render(request, 'prediction/home.html', context)

def predict(request):
    """Handle image upload and prediction"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                # Save the uploaded image with user information
                prediction_result = form.save(commit=False)
                prediction_result.original_filename = request.FILES['uploaded_image'].name
                prediction_result.save()  # This will auto-generate the report_id

                # Get the path to the uploaded image
                image_path = prediction_result.uploaded_image.path

                # Make predictions using all models
                prediction_data = predict_image(image_path)

                # Extract ensemble and individual predictions
                ensemble_pred = prediction_data.get('ensemble_prediction', {})
                individual_preds = prediction_data.get('individual_predictions', {})
                best_model = prediction_data.get('best_model', 'Unknown')

                # Find the best individual model prediction
                best_confidence = 0
                best_class = None

                for model_name, pred_data in individual_preds.items():
                    if pred_data.get('confidence', 0) > best_confidence:
                        best_confidence = pred_data['confidence']
                        best_class = pred_data['predicted_class']

                # Update prediction result with all data
                prediction_result.set_predictions(prediction_data)
                prediction_result.best_model = best_model
                prediction_result.predicted_class = best_class or ensemble_pred.get('predicted_class', 'Unknown')
                prediction_result.confidence = best_confidence or ensemble_pred.get('confidence', 0)

                # Set ensemble results
                prediction_result.ensemble_class = ensemble_pred.get('predicted_class', 'Unknown')
                prediction_result.ensemble_confidence = ensemble_pred.get('confidence', 0)

                prediction_result.save()

                messages.success(request, 'Image analyzed successfully!')
                return redirect('prediction:results', prediction_id=prediction_result.id)

            except Exception as e:
                messages.error(request, f'Error analyzing image: {str(e)}')
                return redirect('prediction:home')
        else:
            messages.error(request, 'Please correct the errors below.')
            return render(request, 'prediction/home.html', {'upload_form': form})

    return redirect('prediction:home')


def search_report(request):
    """Handle report search by ID"""
    if request.method == 'POST':
        form = ReportSearchForm(request.POST)
        if form.is_valid():
            report_id = form.cleaned_data['report_id']
            try:
                prediction = PredictionResult.objects.get(report_id=report_id)
                return redirect('prediction:results', prediction_id=prediction.id)
            except PredictionResult.DoesNotExist:
                messages.error(request, f'No report found with ID: {report_id}')
        else:
            messages.error(request, 'Please enter a valid Report ID.')

    return redirect('prediction:home')

def results(request, prediction_id):
    """Display prediction results"""
    prediction = get_object_or_404(PredictionResult, id=prediction_id)

    # Extract the prediction data
    prediction_data = prediction.predictions

    # Get individual predictions from the ensemble structure
    individual_predictions = prediction_data.get('individual_predictions', {})
    ensemble_prediction = prediction_data.get('ensemble_prediction', {})

    # Add ensemble to the predictions for display
    if ensemble_prediction:
        individual_predictions['Ensemble'] = ensemble_prediction

    context = {
        'prediction': prediction,
        'predictions_data': individual_predictions,
        'ensemble_prediction': ensemble_prediction,
        'best_model': prediction_data.get('best_model', 'Unknown'),
        'models_used': prediction_data.get('models_used', []),
        'classes': ['Bengin', 'Malignant', 'Normal']
    }
    return render(request, 'prediction/results.html', context)
