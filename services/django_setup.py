import os
import sys


def setup_django(*, django_settings: str | None, django_path: str | None) -> None:
    if django_path:
        sys.path.insert(0, django_path)

    if django_settings:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", django_settings)

    import django

    django.setup()
