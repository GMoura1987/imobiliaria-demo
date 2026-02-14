from django.shortcuts import render, get_object_or_404
from .models import Imovel

def index(request):
    imoveis = Imovel.objects.all()
    return render(request, 'core/index.html', {'imoveis': imoveis})

def imovel_detail(request, pk):
    imovel = get_object_or_404(Imovel, pk=pk)
    return render(request, 'core/imovel_detail.html', {'imovel': imovel})
