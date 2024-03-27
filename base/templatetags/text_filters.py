from django import template
from django.utils.safestring import mark_safe
import re

register = template.Library()

@register.filter
def bold_text(value):
    """Converts **text** to <strong>text</strong> and marks the output as safe HTML."""
    value = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', value)
    return mark_safe(value)
