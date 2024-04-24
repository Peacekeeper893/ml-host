from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .ml_nlp import get_local_nlp_recommendations
from .ml_nlp import get_collaborative_filtering_recommendations

# Create your views here.


@api_view(['GET'])
def getRoutes(request):
    print('Hello')
    routes = [
        {
            'Endpoint': '/notes/',
            'method': 'GET',
            'body': None,
            'description': 'Returns an array of notes'
        }
    ]
    return Response(routes)

@api_view(['GET'])
def getSimilar(request):

    name = request.query_params.get('name')
    print(name)
    recommendations = get_local_nlp_recommendations(name)
    print(recommendations)
    return Response(recommendations)

@api_view(['GET'])
def getRecommendations(request):
    name = request.query_params.get('name')
    print(name)
    recommendations = get_collaborative_filtering_recommendations(name)
    print(recommendations)
    return Response(recommendations)
    
    
