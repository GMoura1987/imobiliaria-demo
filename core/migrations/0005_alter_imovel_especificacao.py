from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_imovel_new_fields'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imovel',
            name='especificacao',
            field=models.CharField(blank=True, choices=[('casa', 'Casa'), ('apartamento', 'Apartamento'), ('kitnet', 'Kitnet'), ('comercio', 'Comércio')], max_length=50, null=True),
        ),
    ]
