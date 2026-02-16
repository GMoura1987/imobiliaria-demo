from django.contrib import admin

from .models import Imovel, ImovelImage

class ImovelImageInline(admin.TabularInline):
    model = ImovelImage
    extra = 1

@admin.register(Imovel)
class ImovelAdmin(admin.ModelAdmin):
    inlines = [ImovelImageInline]
    list_display = ('titulo', 'cidade', 'bairro', 'codigo_bairro', 'especificacao', 'preco_aluguel')
    search_fields = ('titulo', 'cidade', 'bairro', 'codigo_bairro')
