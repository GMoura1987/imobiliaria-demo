from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_imovelimage'),
    ]

    operations = [
        migrations.AddField(
            model_name='imovel',
            name='codigo_bairro',
            field=models.CharField(blank=True, help_text='Ex: São Mateus 200', max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='imovel',
            name='especificacao',
            field=models.CharField(blank=True, choices=[('casa', 'Casa'), ('apartamento', 'Apartamento'), ('kitnet', 'Kitnet'), ('galpao', 'Galpão')], max_length=50, null=True),
        ),
    ]
