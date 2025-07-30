from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    return dictionary.get(key)

@register.filter
def get_probability(probabilities, index):
    """Get probability by index and convert to percentage"""
    try:
        return round(probabilities[index] * 100, 1)
    except (IndexError, TypeError):
        return 0

@register.filter
def multiply(value, arg):
    """Multiply value by argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def to_percentage(value):
    """Convert decimal to percentage"""
    try:
        # If value is already > 1, assume it's already a percentage
        if float(value) > 1:
            return round(float(value), 1)
        else:
            return round(float(value) * 100, 1)
    except (ValueError, TypeError):
        return 0
