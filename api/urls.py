from django.urls import path
from api.views import apiView

urlpatterns = [
    path('', apiView.index, name="api-index"),
    path('call', apiView.ApiView.as_view(), name="api-nn-call")
]