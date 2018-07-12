from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from django.shortcuts import render
import json
from django.conf import settings

def index(request):
    return render(request, 'index.html')

class ApiView(APIView):
    renderer_classes = (JSONRenderer, )
    
    def get(self, request, format=None):

        content = {'data': 120}
        return Response(content)
    def post(self, request, format=None):
        from tensorflow.contrib import predictor
        import numpy as np
        import os

        BASE_DIR = getattr(settings, "BASE_DIR", None)
        data = request.data
        print(data)
        predict_fn = predictor.from_saved_model(os.path.join(BASE_DIR, 'exports/1530814024'))
        predictions = predict_fn({"inputs": np.array([[data['data']]])})

        outputs, _ = predictions['scores'].shape
        labels = {0:'negative', 1:'positive'}
        print(predictions)
        for y in range(outputs):
            negative = predictions['scores'][y,0]
            positive = predictions['scores'][y, 1]
            if negative > positive:
                return Response({'prediction': labels[0]})
            else:
                return Response({'prediction': labels[1]})
        return Response({'prediction': 1})