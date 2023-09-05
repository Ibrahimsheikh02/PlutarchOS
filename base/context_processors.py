
from django.conf import settings

#Google Analytics 
def google_analytics(request):
    return {'GA_MEASUREMENT_ID': settings.GA_MEASUREMENT_ID}
