from django.db import models

class Imovel(models.Model):
    titulo = models.CharField(max_length=200)
    descricao = models.TextField()
    quartos = models.IntegerField()
    banheiros = models.IntegerField()
    garagem = models.IntegerField()
    area = models.DecimalField(max_digits=10, decimal_places=2)
    cidade = models.CharField(max_length=100)
    bairro = models.CharField(max_length=100)
    rua = models.CharField(max_length=100)
    numero = models.CharField(max_length=20)
    preco_aluguel = models.DecimalField(max_digits=10, decimal_places=2)
    preco_iptu = models.DecimalField(max_digits=10, decimal_places=2)
    preco_condominio = models.DecimalField(max_digits=10, decimal_places=2)
    aceita_pets = models.BooleanField(default=False)
    imagem = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.titulo

class ImovelImage(models.Model):
    imovel = models.ForeignKey(Imovel, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='imoveis/')

    def __str__(self):
        return f"Image for {self.imovel.titulo}"
