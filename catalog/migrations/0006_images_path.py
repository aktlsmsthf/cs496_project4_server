# Generated by Django 2.0.1 on 2018-01-16 06:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('catalog', '0005_images_label'),
    ]

    operations = [
        migrations.AddField(
            model_name='images',
            name='path',
            field=models.IntegerField(null=True),
        ),
    ]
