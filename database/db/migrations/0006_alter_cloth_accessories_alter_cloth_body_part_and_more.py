# Generated by Django 4.2 on 2024-10-05 07:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0005_cloth_accessories_cloth_body_part_cloth_category_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cloth',
            name='accessories',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='body_part',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='category',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='cloth_orientation',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='coat_length',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='collar',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='color',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='craft',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='elbow',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='fabric',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='hem',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='pant_length',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='pants_contour',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='pattern',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='placket',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='pocket',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='season',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='skirt_length',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='sleeve_length',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='sleeve_type',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='species',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='style',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='suit',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='suitable_age',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='texture',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='type_version',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='upper_dress_contour',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='visible_part',
            field=models.CharField(default='none', max_length=50),
        ),
        migrations.AlterField(
            model_name='cloth',
            name='waist',
            field=models.CharField(default='none', max_length=50),
        ),
    ]